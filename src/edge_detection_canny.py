"""
Canny-based edge detection with morphological processing and contour fitting.

This module implements an alternative edge detection approach that:
1. Uses Canny edge detection with auto-thresholds from Sobel magnitude
2. Applies vertical morphological closing to connect broken edges
3. Extracts connected contours
4. Filters for long, mostly-vertical components near ROI borders
5. Fits smooth curves (RANSAC or spline) to get clean finger boundaries
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy import interpolate

logger = logging.getLogger(__name__)


def detect_edges_canny_contour(
    gradient_data: Dict[str, Any],
    roi_data: Dict[str, Any],
    threshold_low_percentile: float = 30.0,
    threshold_high_percentile: float = 70.0,
    morph_kernel_height: int = 15,
    min_contour_length: int = 50,
    border_search_width: int = 30,
    fit_method: str = "ransac",
    debug_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect finger edges using Canny + morphological processing + contour fitting.

    Pipeline:
    1. Canny edge detection (auto-thresholds from Sobel magnitude)
    2. Vertical morphological closing (connect broken edges)
    3. Extract connected contours
    4. Filter contours (length, orientation, position)
    5. Fit smooth curves to left and right boundaries

    Args:
        gradient_data: Output from apply_sobel_filters()
        roi_data: Output from extract_ring_zone_roi()
        threshold_low_percentile: Percentile for Canny low threshold (0-100)
        threshold_high_percentile: Percentile for Canny high threshold (0-100)
        morph_kernel_height: Height of vertical morphological kernel (pixels)
        min_contour_length: Minimum contour length to keep (pixels)
        border_search_width: Width of border region to search (pixels)
        fit_method: Curve fitting method ("ransac" or "spline")
        debug_dir: Directory to save debug visualizations

    Returns:
        Dictionary containing:
        - left_edges: Array of left edge x-coordinates (one per row)
        - right_edges: Array of right edge x-coordinates (one per row)
        - left_fit: Fitted curve parameters for left edge
        - right_fit: Fitted curve parameters for right edge
        - valid_rows: Boolean array indicating valid edge detections
        - num_valid_rows: Number of rows with valid edges
        - method: "canny_contour"
    """
    gradient_magnitude = gradient_data["gradient_magnitude"]
    roi_image = roi_data["roi_image"]
    h, w = gradient_magnitude.shape

    # Initialize debug observer if needed
    if debug_dir:
        from src.debug_observer import DebugObserver
        observer = DebugObserver(debug_dir)
    else:
        observer = None

    logger.debug(f"Starting Canny-contour edge detection on {w}x{h} ROI")

    # Step 1: Canny edge detection with auto-thresholds
    gradient_uint8 = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Auto-compute thresholds from gradient magnitude distribution
    low_threshold = np.percentile(gradient_magnitude[gradient_magnitude > 0], threshold_low_percentile)
    high_threshold = np.percentile(gradient_magnitude[gradient_magnitude > 0], threshold_high_percentile)

    logger.debug(f"Canny thresholds: low={low_threshold:.1f}, high={high_threshold:.1f}")

    canny_edges = cv2.Canny(gradient_uint8, int(low_threshold), int(high_threshold))

    if observer:
        _save_canny_debug(observer, roi_image, canny_edges, low_threshold, high_threshold, "08a_canny_raw")

    # Step 2: Vertical morphological closing to connect broken edges
    # Use tall, narrow kernel to connect vertically-aligned edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, morph_kernel_height))
    closed_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel)

    if observer:
        _save_morphology_debug(observer, roi_image, canny_edges, closed_edges, kernel.shape, "08b_morph_close")

    # Step 3: Extract connected contours
    contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.debug(f"Found {len(contours)} contours")

    if observer:
        _save_all_contours_debug(observer, roi_image, contours, "08c_all_contours")

    # Step 4: Filter contours
    left_contours = []
    right_contours = []

    for contour in contours:
        # Filter by length
        length = cv2.arcLength(contour, closed=False)
        if length < min_contour_length:
            continue

        # Get bounding box
        x, y, cw, ch = cv2.boundingRect(contour)

        # Check if mostly vertical (height > width)
        if ch <= cw:
            continue

        # Calculate vertical extent (what % of ROI height it spans)
        vertical_extent = ch / h

        # Determine if left or right side based on x position
        is_left = x < (w / 2)
        is_right = (x + cw) > (w / 2)

        # Check if in border regions
        in_left_border = x < border_search_width
        in_right_border = (x + cw) > (w - border_search_width)

        if is_left and in_left_border:
            left_contours.append((contour, length, vertical_extent))
        elif is_right and in_right_border:
            right_contours.append((contour, length, vertical_extent))

    logger.debug(f"Filtered: {len(left_contours)} left contours, {len(right_contours)} right contours")

    # Sort by length (longest first)
    left_contours.sort(key=lambda x: x[1], reverse=True)
    right_contours.sort(key=lambda x: x[1], reverse=True)

    if observer:
        _save_filtered_contours_debug(
            observer, roi_image,
            [c[0] for c in left_contours],
            [c[0] for c in right_contours],
            "08d_filtered_contours"
        )

    # Step 5: Fit smooth curves to boundaries
    left_edges = np.full(h, -1.0, dtype=np.float32)
    right_edges = np.full(h, -1.0, dtype=np.float32)
    valid_rows = np.zeros(h, dtype=bool)

    left_fit = None
    right_fit = None

    # Fit left edge
    if left_contours:
        # Use the longest contour
        best_contour = left_contours[0][0]
        left_edges, left_fit = _fit_contour_curve(best_contour, h, fit_method, side='left')

    # Fit right edge
    if right_contours:
        # Use the longest contour
        best_contour = right_contours[0][0]
        right_edges, right_fit = _fit_contour_curve(best_contour, h, fit_method, side='right')

    # Mark rows as valid if both edges present
    valid_rows = (left_edges >= 0) & (right_edges >= 0)
    num_valid = np.sum(valid_rows)

    logger.debug(f"Valid edges: {num_valid}/{h} rows ({num_valid/h*100:.1f}%)")

    if observer:
        _save_fitted_curves_debug(
            observer, roi_image,
            left_edges, right_edges, valid_rows,
            left_fit, right_fit, fit_method,
            "08e_fitted_curves"
        )

    return {
        "left_edges": left_edges,
        "right_edges": right_edges,
        "left_fit": left_fit,
        "right_fit": right_fit,
        "edge_strengths_left": np.zeros(h),  # Not computed in this method
        "edge_strengths_right": np.zeros(h),  # Not computed in this method
        "valid_rows": valid_rows,
        "num_valid_rows": int(num_valid),
        "filter_orientation": "horizontal",
        "mode_used": "canny_contour",
    }


def _fit_contour_curve(
    contour: np.ndarray,
    roi_height: int,
    method: str,
    side: str
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fit a smooth curve to a contour.

    Args:
        contour: Contour points (Nx1x2 array)
        roi_height: Height of ROI
        method: "ransac" or "spline"
        side: "left" or "right"

    Returns:
        Tuple of (edge_array, fit_params)
    """
    # Extract points
    points = contour.reshape(-1, 2)  # (N, 2) where columns are [x, y]
    y_coords = points[:, 1]
    x_coords = points[:, 0]

    # Initialize edges array
    edges = np.full(roi_height, -1.0, dtype=np.float32)

    if len(points) < 10:
        logger.debug(f"Too few points ({len(points)}) for curve fitting")
        return edges, None

    if method == "ransac":
        # RANSAC polynomial fit (robust to outliers)
        try:
            # Fit 2nd-degree polynomial: x = a*y^2 + b*y + c
            model = make_pipeline(
                PolynomialFeatures(degree=2),
                RANSACRegressor(residual_threshold=5.0, random_state=42)
            )
            model.fit(y_coords.reshape(-1, 1), x_coords)

            # Predict x for all y values
            y_range = np.arange(roi_height).reshape(-1, 1)
            x_pred = model.predict(y_range)

            # Clip to valid range
            x_pred = np.clip(x_pred, 0, None)

            edges[:] = x_pred.flatten()

            # Store fit parameters
            fit_params = {
                "method": "ransac",
                "degree": 2,
                "model": model,
                "inlier_mask": model.named_steps['ransacregressor'].inlier_mask_,
            }

            logger.debug(f"{side} edge: RANSAC fit with {np.sum(fit_params['inlier_mask'])} inliers")

        except Exception as e:
            logger.debug(f"RANSAC fitting failed: {e}")
            fit_params = None

    elif method == "spline":
        # Cubic spline interpolation (smooth curve)
        try:
            # Sort by y coordinate
            sort_idx = np.argsort(y_coords)
            y_sorted = y_coords[sort_idx]
            x_sorted = x_coords[sort_idx]

            # Remove duplicates
            unique_mask = np.concatenate(([True], np.diff(y_sorted) > 0))
            y_unique = y_sorted[unique_mask]
            x_unique = x_sorted[unique_mask]

            if len(y_unique) < 4:
                logger.debug(f"Too few unique points ({len(y_unique)}) for spline")
                return edges, None

            # Fit cubic spline
            spline = interpolate.UnivariateSpline(y_unique, x_unique, k=3, s=len(y_unique))

            # Predict x for all y values
            y_range = np.arange(roi_height)
            x_pred = spline(y_range)

            # Clip to valid range
            x_pred = np.clip(x_pred, 0, None)

            edges[:] = x_pred

            fit_params = {
                "method": "spline",
                "spline": spline,
            }

            logger.debug(f"{side} edge: Spline fit with {len(y_unique)} points")

        except Exception as e:
            logger.debug(f"Spline fitting failed: {e}")
            fit_params = None

    else:
        raise ValueError(f"Unknown fit method: {method}")

    return edges, fit_params


# Debug visualization helpers

def _save_canny_debug(observer, roi_image, canny_edges, low_thresh, high_thresh, filename):
    """Save Canny edge detection debug visualization."""
    from src.viz_constants import FONT_FACE, Color

    vis = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
    vis[canny_edges > 0] = Color.CYAN

    text1 = f"Canny Edges"
    text2 = f"Low: {low_thresh:.1f}, High: {high_thresh:.1f}"
    text3 = f"Edges: {np.sum(canny_edges > 0):,} pixels"

    cv2.putText(vis, text1, (20, 40), FONT_FACE, 1.5, Color.WHITE, 4)
    cv2.putText(vis, text1, (20, 40), FONT_FACE, 1.5, Color.CYAN, 2)

    cv2.putText(vis, text2, (20, 80), FONT_FACE, 1.0, Color.WHITE, 3)
    cv2.putText(vis, text2, (20, 80), FONT_FACE, 1.0, Color.YELLOW, 2)

    cv2.putText(vis, text3, (20, 120), FONT_FACE, 1.0, Color.WHITE, 3)
    cv2.putText(vis, text3, (20, 120), FONT_FACE, 1.0, Color.GREEN, 2)

    observer.save_stage(filename, vis)


def _save_morphology_debug(observer, roi_image, before, after, kernel_shape, filename):
    """Save morphological operation debug visualization."""
    from src.viz_constants import FONT_FACE, Color

    h, w = roi_image.shape
    vis = np.zeros((h, w * 2, 3), dtype=np.uint8)

    # Left: before
    left = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
    left[before > 0] = Color.CYAN
    vis[:, :w] = left

    # Right: after
    right = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
    right[after > 0] = Color.GREEN
    vis[:, w:] = right

    # Labels
    cv2.putText(vis, "Before Closing", (20, 40), FONT_FACE, 1.5, Color.WHITE, 4)
    cv2.putText(vis, "Before Closing", (20, 40), FONT_FACE, 1.5, Color.CYAN, 2)

    cv2.putText(vis, "After Closing", (w + 20, 40), FONT_FACE, 1.5, Color.WHITE, 4)
    cv2.putText(vis, "After Closing", (w + 20, 40), FONT_FACE, 1.5, Color.GREEN, 2)

    text = f"Kernel: {kernel_shape[0]}x{kernel_shape[1]} (WxH)"
    cv2.putText(vis, text, (w + 20, 80), FONT_FACE, 1.0, Color.WHITE, 3)
    cv2.putText(vis, text, (w + 20, 80), FONT_FACE, 1.0, Color.YELLOW, 2)

    observer.save_stage(filename, vis)


def _save_all_contours_debug(observer, roi_image, contours, filename):
    """Save all detected contours visualization."""
    from src.viz_constants import FONT_FACE, Color

    vis = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, contours, -1, Color.CYAN, 2)

    text1 = f"All Contours: {len(contours)}"
    cv2.putText(vis, text1, (20, 40), FONT_FACE, 1.5, Color.WHITE, 4)
    cv2.putText(vis, text1, (20, 40), FONT_FACE, 1.5, Color.CYAN, 2)

    observer.save_stage(filename, vis)


def _save_filtered_contours_debug(observer, roi_image, left_contours, right_contours, filename):
    """Save filtered contours visualization."""
    from src.viz_constants import FONT_FACE, Color

    vis = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)

    # Draw left contours in cyan
    cv2.drawContours(vis, left_contours, -1, Color.CYAN, 3)

    # Draw right contours in magenta
    cv2.drawContours(vis, right_contours, -1, Color.MAGENTA, 3)

    text1 = f"Filtered Contours"
    text2 = f"Left: {len(left_contours)}, Right: {len(right_contours)}"

    cv2.putText(vis, text1, (20, 40), FONT_FACE, 1.5, Color.WHITE, 4)
    cv2.putText(vis, text1, (20, 40), FONT_FACE, 1.5, Color.GREEN, 2)

    cv2.putText(vis, text2, (20, 80), FONT_FACE, 1.0, Color.WHITE, 3)
    cv2.putText(vis, text2, (20, 80), FONT_FACE, 1.0, Color.YELLOW, 2)

    observer.save_stage(filename, vis)


def _save_fitted_curves_debug(observer, roi_image, left_edges, right_edges, valid_rows,
                               left_fit, right_fit, fit_method, filename):
    """Save fitted curves visualization."""
    from src.viz_constants import FONT_FACE, Color

    vis = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)

    # Draw fitted curves
    for y in range(len(left_edges)):
        if valid_rows[y]:
            left_x = int(left_edges[y])
            right_x = int(right_edges[y])

            cv2.circle(vis, (left_x, y), 2, Color.CYAN, -1)
            cv2.circle(vis, (right_x, y), 2, Color.MAGENTA, -1)

    text1 = f"Fitted Curves ({fit_method.upper()})"
    text2 = f"Valid: {np.sum(valid_rows)}/{len(valid_rows)} rows"

    cv2.putText(vis, text1, (20, 40), FONT_FACE, 1.5, Color.WHITE, 4)
    cv2.putText(vis, text1, (20, 40), FONT_FACE, 1.5, Color.GREEN, 2)

    cv2.putText(vis, text2, (20, 80), FONT_FACE, 1.0, Color.WHITE, 3)
    cv2.putText(vis, text2, (20, 80), FONT_FACE, 1.0, Color.YELLOW, 2)

    # Add legend
    h = vis.shape[0]
    legend_y = h - 80
    cv2.putText(vis, "Cyan: Left edge", (20, legend_y), FONT_FACE, 1.0, Color.CYAN, 2)
    cv2.putText(vis, "Magenta: Right edge", (20, legend_y + 30), FONT_FACE, 1.0, Color.MAGENTA, 2)

    observer.save_stage(filename, vis)
