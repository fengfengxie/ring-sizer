"""
Edge refinement using Sobel gradient filtering.

This module implements v1's core innovation: replacing contour-based width
measurement with gradient-based edge detection for improved accuracy.

Functions:
- extract_ring_zone_roi: Extract ROI around ring zone
- apply_sobel_filters: Bidirectional Sobel filtering
- detect_edges_per_row: Find left/right edges in each cross-section
- measure_width_from_edges: Compute width from edge positions
- compute_edge_quality_score: Assess edge detection quality (Phase 3)
- refine_edges_sobel: Main entry point for edge refinement
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List


def extract_ring_zone_roi(
    image: np.ndarray,
    axis_data: Dict[str, Any],
    zone_data: Dict[str, Any],
    finger_mask: Optional[np.ndarray] = None,
    padding: int = 50,
    rotate_align: bool = False
) -> Dict[str, Any]:
    """
    Extract rectangular ROI around ring zone.

    Args:
        image: Input BGR image
        axis_data: Output from estimate_finger_axis()
        zone_data: Output from localize_ring_zone()
        padding: Extra pixels around zone for gradient context (default 50)
        rotate_align: If True, rotate ROI so finger axis is vertical

    Returns:
        Dictionary containing:
        - roi_image: Extracted ROI (grayscale)
        - roi_mask: Extracted finger mask ROI (if finger_mask provided)
        - roi_bounds: (x_min, y_min, x_max, y_max) in original image
        - transform_matrix: 3x3 matrix to map ROI coords -> original coords
        - inverse_transform: 3x3 matrix to map original -> ROI coords
        - rotation_angle: Rotation angle applied (degrees)
        - roi_width: ROI width in pixels
        - roi_height: ROI height in pixels
    """
    h, w = image.shape[:2]

    # Extract zone information
    start_point = zone_data["start_point"]
    end_point = zone_data["end_point"]
    direction = axis_data["direction"]

    # Perpendicular direction
    perp_direction = np.array([-direction[1], direction[0]], dtype=np.float32)

    # Calculate zone length and estimated width
    zone_length = zone_data["length"]
    # Estimate finger width from axis length (rough approximation)
    # Typical finger aspect ratio is ~3:1 to 5:1 (length:width)
    # Use conservative estimate to ensure we capture full finger width
    estimated_width = axis_data["length"] / 3.0  # More conservative

    # Define ROI bounds
    # ROI extends from start to end along axis, ±(width/2 + padding) perpendicular
    half_width = (estimated_width / 2.0) + padding

    # Calculate corner points of ROI
    # Start from zone start, extend perpendicular in both directions
    roi_corners = []
    for axis_pt in [start_point, end_point]:
        # Extend padding along axis direction (for gradient context)
        for perp_sign in [-1, 1]:
            corner = axis_pt + perp_direction * (half_width * perp_sign)
            roi_corners.append(corner)

    # Add axis padding
    axis_padding = padding
    start_extended = start_point - direction * axis_padding
    end_extended = end_point + direction * axis_padding

    # Recalculate corners with axis padding
    roi_corners = []
    for axis_pt in [start_extended, end_extended]:
        for perp_sign in [-1, 1]:
            corner = axis_pt + perp_direction * (half_width * perp_sign)
            roi_corners.append(corner)

    roi_corners = np.array(roi_corners, dtype=np.float32)

    # Calculate bounding box
    x_coords = roi_corners[:, 0]
    y_coords = roi_corners[:, 1]

    x_min = int(np.clip(np.min(x_coords), 0, w - 1))
    x_max = int(np.clip(np.max(x_coords), 0, w - 1))
    y_min = int(np.clip(np.min(y_coords), 0, h - 1))
    y_max = int(np.clip(np.max(y_coords), 0, h - 1))

    roi_width = x_max - x_min
    roi_height = y_max - y_min

    if roi_width < 10 or roi_height < 10:
        raise ValueError(f"ROI too small: {roi_width}x{roi_height}")

    # Extract ROI
    roi_bgr = image[y_min:y_max, x_min:x_max].copy()

    # Convert to grayscale for edge detection
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Extract finger mask ROI if provided
    roi_mask = None
    if finger_mask is not None:
        roi_mask = finger_mask[y_min:y_max, x_min:x_max].copy()

    # Create transform matrix (ROI coords -> original coords)
    # Simple translation for non-rotated case
    transform = np.eye(3, dtype=np.float32)
    transform[0, 2] = x_min  # Translation in x
    transform[1, 2] = y_min  # Translation in y

    inverse_transform = np.linalg.inv(transform)

    rotation_angle = 0.0

    # Optional rotation alignment
    if rotate_align:
        # Calculate rotation angle to make finger vertical
        # Finger direction -> make it point upward (0, -1)
        # Current direction is (dx, dy), want to rotate to (0, -1)
        rotation_angle = np.degrees(np.arctan2(-direction[0], direction[1]))

        # Get rotation matrix
        roi_center = (roi_width / 2.0, roi_height / 2.0)
        rotation_matrix = cv2.getRotationMatrix2D(roi_center, rotation_angle, 1.0)

        # Rotate ROI
        roi_gray = cv2.warpAffine(
            roi_gray, rotation_matrix, (roi_width, roi_height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

        # Update transform matrices
        # Rotation matrix is 2x3, convert to 3x3 for composition
        rotation_matrix_3x3 = np.eye(3, dtype=np.float32)
        rotation_matrix_3x3[:2, :] = rotation_matrix

        # Compose: translate then rotate
        transform = np.dot(rotation_matrix_3x3, transform)
        inverse_transform = np.linalg.inv(transform)

    return {
        "roi_image": roi_gray,
        "roi_mask": roi_mask,  # Finger mask ROI
        "roi_bgr": roi_bgr,  # Keep BGR for debug visualization
        "roi_bounds": (x_min, y_min, x_max, y_max),
        "transform_matrix": transform,
        "inverse_transform": inverse_transform,
        "rotation_angle": rotation_angle,
        "roi_width": roi_width,
        "roi_height": roi_height,
        "zone_start_in_roi": start_point - np.array([x_min, y_min], dtype=np.float32),
        "zone_end_in_roi": end_point - np.array([x_min, y_min], dtype=np.float32),
    }


def apply_sobel_filters(
    roi_image: np.ndarray,
    kernel_size: int = 3,
    axis_direction: str = "auto"
) -> Dict[str, Any]:
    """
    Apply bidirectional Sobel filters to detect edges.

    For vertical finger (axis_direction="vertical"):
    - Use horizontal Sobel kernels (detect left/right edges)

    For horizontal finger (axis_direction="horizontal"):
    - Use vertical Sobel kernels (detect top/bottom edges)

    Auto mode detects orientation from ROI aspect ratio.

    Args:
        roi_image: Grayscale ROI image
        kernel_size: Sobel kernel size (3, 5, or 7)
        axis_direction: Finger axis direction ("auto", "vertical", "horizontal")

    Returns:
        Dictionary containing:
        - gradient_x: Horizontal gradient (Sobel X)
        - gradient_y: Vertical gradient (Sobel Y)
        - gradient_magnitude: Combined gradient magnitude
        - gradient_direction: Edge orientation (radians)
        - kernel_size: Kernel size used
        - filter_orientation: "horizontal" or "vertical"
    """
    if kernel_size not in [3, 5, 7]:
        raise ValueError(f"Invalid kernel_size: {kernel_size}. Use 3, 5, or 7")

    h, w = roi_image.shape

    # Determine filter orientation
    if axis_direction == "auto":
        # If ROI is wider than tall, finger is likely horizontal
        if w > h:
            filter_orientation = "vertical"  # Detect top/bottom edges
        else:
            filter_orientation = "horizontal"  # Detect left/right edges
    elif axis_direction == "vertical":
        filter_orientation = "horizontal"
    elif axis_direction == "horizontal":
        filter_orientation = "vertical"
    else:
        raise ValueError(f"Invalid axis_direction: {axis_direction}")

    # Apply Sobel filters
    # Sobel X detects vertical edges (left/right boundaries)
    # Sobel Y detects horizontal edges (top/bottom boundaries)

    # Use cv2.Sobel for standard implementation
    grad_x = cv2.Sobel(roi_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(roi_image, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Calculate gradient direction (angle)
    gradient_direction = np.arctan2(grad_y, grad_x)

    # Normalize gradients to 0-255 for visualization
    grad_x_normalized = np.clip(np.abs(grad_x), 0, 255).astype(np.uint8)
    grad_y_normalized = np.clip(np.abs(grad_y), 0, 255).astype(np.uint8)
    grad_mag_normalized = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

    return {
        "gradient_x": grad_x,
        "gradient_y": grad_y,
        "gradient_magnitude": gradient_magnitude,
        "gradient_direction": gradient_direction,
        "gradient_x_normalized": grad_x_normalized,
        "gradient_y_normalized": grad_y_normalized,
        "gradient_mag_normalized": grad_mag_normalized,
        "kernel_size": kernel_size,
        "filter_orientation": filter_orientation,
    }


def detect_edges_per_row(
    gradient_data: Dict[str, Any],
    roi_data: Dict[str, Any],
    threshold: float = 30.0,
    expected_width_px: Optional[float] = None
) -> Dict[str, Any]:
    """
    Detect left and right finger edges for each row (cross-section).

    For each row in the ROI:
    1. Find all pixels above gradient threshold
    2. Separate into left candidates (x < center) and right (x > center)
    3. Select leftmost strong edge as left boundary
    4. Select rightmost strong edge as right boundary
    5. Validate: width reasonable, edges symmetric

    Args:
        gradient_data: Output from apply_sobel_filters()
        roi_data: Output from extract_ring_zone_roi()
        threshold: Minimum gradient magnitude for valid edge
        expected_width_px: Expected finger width (optional, for validation)

    Returns:
        Dictionary containing:
        - left_edges: Array of left edge x-coordinates (one per row)
        - right_edges: Array of right edge x-coordinates (one per row)
        - edge_strengths_left: Gradient magnitude at left edges
        - edge_strengths_right: Gradient magnitude at right edges
        - valid_rows: Boolean mask of rows with successful detection
        - num_valid_rows: Count of successful detections
    """
    gradient_magnitude = gradient_data["gradient_magnitude"]
    filter_orientation = gradient_data["filter_orientation"]

    h, w = gradient_magnitude.shape

    # Use mask to constrain edge search if available
    use_mask = roi_data.get("roi_mask") is not None
    roi_mask = roi_data.get("roi_mask") if use_mask else None

    # For horizontal filter orientation (detecting left/right edges)
    # Process each row to find left and right edges
    if filter_orientation == "horizontal":
        num_rows = h
        left_edges = np.full(num_rows, -1.0, dtype=np.float32)
        right_edges = np.full(num_rows, -1.0, dtype=np.float32)
        edge_strengths_left = np.zeros(num_rows, dtype=np.float32)
        edge_strengths_right = np.zeros(num_rows, dtype=np.float32)
        valid_rows = np.zeros(num_rows, dtype=bool)

        for row in range(num_rows):
            # If mask available, find leftmost and rightmost mask pixels
            if use_mask:
                row_mask = roi_mask[row, :]
                mask_pixels = np.where(row_mask > 0)[0]

                if len(mask_pixels) < 10:  # Need reasonable finger width
                    continue

                # Finger boundaries are at the edges of the mask
                mask_left = mask_pixels[0]
                mask_right = mask_pixels[-1]

                # Look for strong gradients near mask boundaries (within ±10px)
                search_radius = 10
                left_search_start = max(0, mask_left - search_radius)
                left_search_end = min(w, mask_left + search_radius)
                right_search_start = max(0, mask_right - search_radius)
                right_search_end = min(w, mask_right + search_radius)

                row_gradient = gradient_magnitude[row, :]

                # Find strongest edge in left search region
                left_region_grad = row_gradient[left_search_start:left_search_end]
                if len(left_region_grad) > 0:
                    left_rel_idx = np.argmax(left_region_grad)
                    left_edge_x = left_search_start + left_rel_idx
                    left_strength = left_region_grad[left_rel_idx]
                else:
                    left_edge_x = mask_left
                    left_strength = row_gradient[mask_left]

                # Find strongest edge in right search region
                right_region_grad = row_gradient[right_search_start:right_search_end]
                if len(right_region_grad) > 0:
                    right_rel_idx = np.argmax(right_region_grad)
                    right_edge_x = right_search_start + right_rel_idx
                    right_strength = right_region_grad[right_rel_idx]
                else:
                    right_edge_x = mask_right
                    right_strength = row_gradient[mask_right]

            else:
                # Original gradient-only method
                row_gradient = gradient_magnitude[row, :]
                strong_edges = np.where(row_gradient > threshold)[0]

                if len(strong_edges) < 2:
                    continue

                roi_center_x = w / 2.0
                left_candidates = strong_edges[strong_edges < roi_center_x]
                right_candidates = strong_edges[strong_edges >= roi_center_x]

                if len(left_candidates) == 0 or len(right_candidates) == 0:
                    continue

                # Take leftmost and rightmost edges (outermost boundaries)
                left_edge_x = left_candidates[0]
                right_edge_x = right_candidates[-1]

                left_strength = row_gradient[left_edge_x]
                right_strength = row_gradient[right_edge_x]

            # Calculate width
            width = right_edge_x - left_edge_x

            # Validate width if expected width provided
            if expected_width_px is not None:
                if width < expected_width_px * 0.5 or width > expected_width_px * 1.5:
                    continue

            # Mark as valid
            left_edges[row] = float(left_edge_x)
            right_edges[row] = float(right_edge_x)
            edge_strengths_left[row] = left_strength
            edge_strengths_right[row] = right_strength
            valid_rows[row] = True

    else:
        # Vertical filter orientation (detecting top/bottom edges)
        # Process each column
        num_cols = w
        left_edges = np.full(num_cols, -1.0, dtype=np.float32)
        right_edges = np.full(num_cols, -1.0, dtype=np.float32)
        edge_strengths_left = np.zeros(num_cols, dtype=np.float32)
        edge_strengths_right = np.zeros(num_cols, dtype=np.float32)
        valid_rows = np.zeros(num_cols, dtype=bool)

        roi_center_y = h / 2.0

        for col in range(num_cols):
            col_gradient = gradient_magnitude[:, col]

            strong_edges = np.where(col_gradient > threshold)[0]

            if len(strong_edges) < 2:
                continue

            top_candidates = strong_edges[strong_edges < roi_center_y]
            bottom_candidates = strong_edges[strong_edges >= roi_center_y]

            if len(top_candidates) == 0 or len(bottom_candidates) == 0:
                continue

            # Select edges closest to center (finger boundaries)
            top_edge_y = top_candidates[-1]  # Bottommost of top candidates
            bottom_edge_y = bottom_candidates[0]  # Topmost of bottom candidates

            top_strength = col_gradient[top_edge_y]
            bottom_strength = col_gradient[bottom_edge_y]

            height = bottom_edge_y - top_edge_y

            if expected_width_px is not None:
                if height < expected_width_px * 0.5 or height > expected_width_px * 1.5:
                    continue

            left_edges[col] = float(top_edge_y)
            right_edges[col] = float(bottom_edge_y)
            edge_strengths_left[col] = top_strength
            edge_strengths_right[col] = bottom_strength
            valid_rows[col] = True

    num_valid = np.sum(valid_rows)

    return {
        "left_edges": left_edges,
        "right_edges": right_edges,
        "edge_strengths_left": edge_strengths_left,
        "edge_strengths_right": edge_strengths_right,
        "valid_rows": valid_rows,
        "num_valid_rows": int(num_valid),
        "filter_orientation": filter_orientation,
    }


def measure_width_from_edges(
    edge_data: Dict[str, Any],
    roi_data: Dict[str, Any],
    scale_px_per_cm: float
) -> Dict[str, Any]:
    """
    Compute finger width from detected edges.

    Steps:
    1. Calculate width for each valid row: width_px = right_edge - left_edge
    2. Filter outliers (>2 std dev from median)
    3. Compute statistics (median, mean, std)
    4. Convert width from pixels to cm

    Args:
        edge_data: Output from detect_edges_per_row()
        roi_data: Output from extract_ring_zone_roi()
        scale_px_per_cm: Pixels per cm from card detection

    Returns:
        Dictionary containing:
        - widths_px: Array of width measurements (pixels)
        - median_width_px: Median width in pixels
        - median_width_cm: Median width in cm (final measurement)
        - mean_width_px: Mean width in pixels
        - std_width_px: Standard deviation of widths
        - num_samples: Number of valid width measurements
        - outliers_removed: Number of outliers filtered
    """
    left_edges = edge_data["left_edges"]
    right_edges = edge_data["right_edges"]
    valid_rows = edge_data["valid_rows"]

    # Calculate widths for valid rows
    widths_px = []
    for i in range(len(valid_rows)):
        if valid_rows[i]:
            width = right_edges[i] - left_edges[i]
            if width > 0:
                widths_px.append(width)

    if len(widths_px) == 0:
        raise ValueError("No valid width measurements found")

    widths_px = np.array(widths_px)

    # Filter outliers using median absolute deviation (MAD)
    median = np.median(widths_px)
    mad = np.median(np.abs(widths_px - median))

    # Outliers are >3 MAD from median (more robust than std dev)
    if mad > 0:
        outlier_threshold = 3.0
        is_outlier = np.abs(widths_px - median) > (outlier_threshold * mad)
        widths_filtered = widths_px[~is_outlier]
        outliers_removed = np.sum(is_outlier)
    else:
        widths_filtered = widths_px
        outliers_removed = 0

    if len(widths_filtered) == 0:
        # All measurements were outliers, use original
        widths_filtered = widths_px
        outliers_removed = 0

    # Calculate statistics
    median_width_px = float(np.median(widths_filtered))
    mean_width_px = float(np.mean(widths_filtered))
    std_width_px = float(np.std(widths_filtered))

    # Convert to cm
    median_width_cm = median_width_px / scale_px_per_cm

    return {
        "widths_px": widths_filtered.tolist(),
        "median_width_px": median_width_px,
        "median_width_cm": median_width_cm,
        "mean_width_px": mean_width_px,
        "std_width_px": std_width_px,
        "num_samples": len(widths_filtered),
        "outliers_removed": int(outliers_removed),
    }


def refine_edges_sobel(
    image: np.ndarray,
    axis_data: Dict[str, Any],
    zone_data: Dict[str, Any],
    scale_px_per_cm: float,
    finger_mask: Optional[np.ndarray] = None,
    sobel_threshold: float = 15.0,
    kernel_size: int = 3,
    rotate_align: bool = False,
    expected_width_px: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Main entry point for Sobel-based edge refinement.

    Replaces contour-based width measurement with gradient-based edge detection.

    Pipeline:
    1. Extract ROI around ring zone
    2. Apply bidirectional Sobel filters
    3. Detect left/right edges per cross-section
    4. Measure width from edges
    5. Convert to cm and return measurement

    Args:
        image: Input BGR image
        axis_data: Output from estimate_finger_axis()
        zone_data: Output from localize_ring_zone()
        scale_px_per_cm: Pixels per cm from card detection
        sobel_threshold: Minimum gradient magnitude for valid edge
        kernel_size: Sobel kernel size (3, 5, or 7)
        rotate_align: Rotate ROI for vertical finger alignment
        expected_width_px: Expected width for validation (optional)

    Returns:
        Dictionary containing:
        - median_width_cm: Final measurement in cm
        - median_width_px: Measurement in pixels
        - std_width_px: Standard deviation
        - num_samples: Number of valid measurements
        - edge_detection_success_rate: % of rows with valid edges
        - roi_data: ROI extraction data
        - gradient_data: Sobel filter data
        - edge_data: Edge detection data
        - method: "sobel"
    """
    # Step 1: Extract ROI
    roi_data = extract_ring_zone_roi(
        image, axis_data, zone_data,
        finger_mask=finger_mask,
        padding=50, rotate_align=rotate_align
    )

    # Step 2: Apply Sobel filters
    gradient_data = apply_sobel_filters(
        roi_data["roi_image"],
        kernel_size=kernel_size,
        axis_direction="auto"
    )

    # Step 3: Detect edges per row
    edge_data = detect_edges_per_row(
        gradient_data, roi_data,
        threshold=sobel_threshold,
        expected_width_px=expected_width_px
    )

    # Step 4: Measure width from edges
    width_data = measure_width_from_edges(
        edge_data, roi_data, scale_px_per_cm
    )

    # Calculate success rate
    total_rows = len(edge_data["valid_rows"])
    success_rate = edge_data["num_valid_rows"] / total_rows if total_rows > 0 else 0.0

    # Combine results
    return {
        "median_width_cm": width_data["median_width_cm"],
        "median_width_px": width_data["median_width_px"],
        "mean_width_px": width_data["mean_width_px"],
        "std_width_px": width_data["std_width_px"],
        "num_samples": width_data["num_samples"],
        "outliers_removed": width_data["outliers_removed"],
        "edge_detection_success_rate": success_rate,
        "roi_data": roi_data,
        "gradient_data": gradient_data,
        "edge_data": edge_data,
        "method": "sobel",
    }
