"""
Edge refinement using Sobel gradient filtering.

This module implements v1's core innovation: replacing contour-based width
measurement with gradient-based edge detection for improved accuracy.

Functions:
- extract_ring_zone_roi: Extract ROI around ring zone
- apply_sobel_filters: Bidirectional Sobel filtering
- detect_edges_per_row: Find left/right edges in each cross-section
- refine_edge_subpixel: Sub-pixel edge localization (Phase 3)
- measure_width_from_edges: Compute width from edge positions
- compute_edge_quality_score: Assess edge detection quality (Phase 3)
- should_use_sobel_measurement: Auto fallback logic (Phase 3)
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

    # Convert axis center point and direction to ROI coordinates
    axis_center = axis_data.get("center", (start_point + end_point) / 2)
    axis_center_in_roi = axis_center - np.array([x_min, y_min], dtype=np.float32)
    
    # Direction vector stays the same (it's not affected by translation)
    axis_direction_in_roi = direction.copy()

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
        "axis_center_in_roi": axis_center_in_roi,  # NEW: axis center in ROI coords
        "axis_direction_in_roi": axis_direction_in_roi,  # NEW: axis direction vector
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
        # After rotation normalization, finger is always vertical (upright)
        # Finger runs vertically → detect left/right edges → use horizontal filter
        # 
        # NOTE: ROI aspect ratio is NOT reliable after rotation normalization!
        # The ROI may be wider than tall even when finger is vertical.
        # Always use horizontal filter orientation for upright hands.
        filter_orientation = "horizontal"  # Detect left/right edges for vertical finger
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
    2. Use finger axis to determine left vs right side
    3. Select strongest edge on left side of axis as left boundary
    4. Select strongest edge on right side of axis as right boundary
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
    
    # Get axis information for left/right determination
    axis_center = roi_data.get("axis_center_in_roi")
    axis_direction = roi_data.get("axis_direction_in_roi")
    zone_start = roi_data.get("zone_start_in_roi")
    zone_end = roi_data.get("zone_end_in_roi")
    
    # Helper function: determine which side of axis a point is on
    def which_side_of_axis(point_x, point_y):
        """
        Determine if point is left or right of finger axis.
        Returns: -1 for left, +1 for right, 0 on axis
        
        Uses cross product: if (point - axis_point) × axis_direction > 0, point is on left
        """
        if axis_center is None or axis_direction is None:
            # Fallback to simple center-based logic
            return -1 if point_x < w / 2 else 1
        
        # Find closest point on axis line to current row
        # Axis passes through axis_center with direction axis_direction
        # For row y, find x-coordinate of axis at this y
        
        if abs(axis_direction[1]) < 1e-6:
            # Nearly horizontal axis (shouldn't happen with upright hand)
            axis_x_at_row = axis_center[0]
        else:
            # Parametric line: P = axis_center + t * axis_direction
            # At row y: axis_center[1] + t * axis_direction[1] = point_y
            t = (point_y - axis_center[1]) / axis_direction[1]
            axis_x_at_row = axis_center[0] + t * axis_direction[0]
        
        # Determine side using cross product
        # Vector from axis point to test point
        to_point = np.array([point_x - axis_x_at_row, 0], dtype=np.float32)
        # Perpendicular to axis (pointing left)
        perp_left = np.array([-axis_direction[1], axis_direction[0]], dtype=np.float32)
        
        # Dot product tells us which side
        side_value = np.dot(to_point, perp_left)
        
        if side_value > 0:
            return -1  # Left side
        elif side_value < 0:
            return 1   # Right side
        else:
            return 0   # On axis

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

                # NEW: Validate edges are on correct side of axis
                # Find strongest edge in left search region (must be on left side of axis)
                left_edge_x = None
                left_strength = 0
                left_region_grad = row_gradient[left_search_start:left_search_end]
                if len(left_region_grad) > 0:
                    # Find all peaks in left region
                    for i in range(len(left_region_grad)):
                        x = left_search_start + i
                        if left_region_grad[i] > threshold:
                            # Check if this edge is on the left side of axis
                            side = which_side_of_axis(x, row)
                            if side <= 0:  # Left or on axis
                                if left_region_grad[i] > left_strength:
                                    left_edge_x = x
                                    left_strength = left_region_grad[i]
                
                if left_edge_x is None:
                    # Fallback: use mask boundary if no valid edge found
                    left_edge_x = mask_left
                    left_strength = row_gradient[mask_left]

                # NEW: Find strongest edge in right search region (must be on right side of axis)
                right_edge_x = None
                right_strength = 0
                right_region_grad = row_gradient[right_search_start:right_search_end]
                if len(right_region_grad) > 0:
                    # Find all peaks in right region
                    for i in range(len(right_region_grad)):
                        x = right_search_start + i
                        if right_region_grad[i] > threshold:
                            # Check if this edge is on the right side of axis
                            side = which_side_of_axis(x, row)
                            if side >= 0:  # Right or on axis
                                if right_region_grad[i] > right_strength:
                                    right_edge_x = x
                                    right_strength = right_region_grad[i]
                
                if right_edge_x is None:
                    # Fallback: use mask boundary if no valid edge found
                    right_edge_x = mask_right
                    right_strength = row_gradient[mask_right]

            else:
                # Gradient-only method with axis constraint
                row_gradient = gradient_magnitude[row, :]
                strong_edges = np.where(row_gradient > threshold)[0]

                if len(strong_edges) < 2:
                    continue

                # NEW: Separate edges by which side of axis they're on
                left_candidates = []
                right_candidates = []
                
                for edge_x in strong_edges:
                    side = which_side_of_axis(edge_x, row)
                    if side < 0:  # Left side
                        left_candidates.append(edge_x)
                    elif side > 0:  # Right side
                        right_candidates.append(edge_x)
                    # Edges exactly on axis are ignored

                if len(left_candidates) == 0 or len(right_candidates) == 0:
                    continue

                # Take leftmost left edge and rightmost right edge (outermost boundaries)
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


def refine_edge_subpixel(
    gradient_magnitude: np.ndarray,
    edge_positions: np.ndarray,
    valid_mask: np.ndarray,
    method: str = "parabola"
) -> np.ndarray:
    """
    Refine edge positions to sub-pixel precision.

    Uses parabola fitting on gradient magnitude to find peak position
    with <0.5 pixel accuracy.

    Args:
        gradient_magnitude: 2D gradient magnitude array
        edge_positions: Integer edge positions (one per row/col)
        valid_mask: Boolean mask indicating which positions are valid
        method: Refinement method ("parabola" or "gaussian")

    Returns:
        Refined edge positions (float, sub-pixel precision)
    """
    refined_positions = edge_positions.copy()

    if method == "parabola":
        # Parabola fitting: fit f(x) = ax^2 + bx + c to 3 points
        # Peak at x = -b/(2a)

        for i in range(len(edge_positions)):
            if not valid_mask[i]:
                continue

            edge_pos = int(edge_positions[i])

            # Get gradient magnitude at edge and neighbors
            # Handle edge cases (pun intended)
            if edge_pos <= 0 or edge_pos >= gradient_magnitude.shape[1] - 1:
                continue  # Can't refine at image boundaries

            # For horizontal orientation (row-wise edge detection)
            if len(gradient_magnitude.shape) == 2 and i < gradient_magnitude.shape[0]:
                # Sample gradient at x-1, x, x+1
                x_minus = edge_pos - 1
                x_center = edge_pos
                x_plus = edge_pos + 1

                g_minus = gradient_magnitude[i, x_minus]
                g_center = gradient_magnitude[i, x_center]
                g_plus = gradient_magnitude[i, x_plus]

                # Fit parabola: f(x) = ax^2 + bx + c
                # Using x = -1, 0, 1 for simplicity
                # f(-1) = a - b + c = g_minus
                # f(0) = c = g_center
                # f(1) = a + b + c = g_plus

                c = g_center
                a = (g_plus + g_minus - 2 * c) / 2.0
                b = (g_plus - g_minus) / 2.0

                # Peak at x_peak = -b/(2a)
                if abs(a) > 1e-6:  # Avoid division by zero
                    x_peak = -b / (2.0 * a)

                    # Constrain to reasonable range (±0.5 pixels)
                    if abs(x_peak) <= 0.5:
                        refined_positions[i] = edge_pos + x_peak

    elif method == "gaussian":
        # Gaussian fitting (more complex, not implemented yet)
        # Would fit Gaussian to 5-pixel window
        # For now, fall back to parabola
        return refine_edge_subpixel(gradient_magnitude, edge_positions, valid_mask, method="parabola")

    else:
        raise ValueError(f"Unknown refinement method: {method}")

    return refined_positions


def measure_width_from_edges(
    edge_data: Dict[str, Any],
    roi_data: Dict[str, Any],
    scale_px_per_cm: float,
    gradient_data: Optional[Dict[str, Any]] = None,
    use_subpixel: bool = True
) -> Dict[str, Any]:
    """
    Compute finger width from detected edges.

    Steps:
    1. Apply sub-pixel refinement if gradient data available
    2. Calculate width for each valid row: width_px = right_edge - left_edge
    3. Filter outliers (>3 MAD from median)
    4. Compute statistics (median, mean, std)
    5. Convert width from pixels to cm

    Args:
        edge_data: Output from detect_edges_per_row()
        roi_data: Output from extract_ring_zone_roi()
        scale_px_per_cm: Pixels per cm from card detection
        gradient_data: Optional gradient data for sub-pixel refinement
        use_subpixel: Enable sub-pixel refinement (default True)

    Returns:
        Dictionary containing:
        - widths_px: Array of width measurements (pixels)
        - median_width_px: Median width in pixels
        - median_width_cm: Median width in cm (final measurement)
        - mean_width_px: Mean width in pixels
        - std_width_px: Standard deviation of widths
        - num_samples: Number of valid width measurements
        - outliers_removed: Number of outliers filtered
        - subpixel_refinement_used: Whether sub-pixel refinement was applied
    """
    left_edges = edge_data["left_edges"].copy()
    right_edges = edge_data["right_edges"].copy()
    valid_rows = edge_data["valid_rows"]

    # Apply sub-pixel refinement if available
    subpixel_used = False
    if use_subpixel and gradient_data is not None:
        try:
            gradient_magnitude = gradient_data["gradient_magnitude"]

            # Refine left edges
            left_edges = refine_edge_subpixel(
                gradient_magnitude, left_edges, valid_rows, method="parabola"
            )

            # Refine right edges
            right_edges = refine_edge_subpixel(
                gradient_magnitude, right_edges, valid_rows, method="parabola"
            )

            subpixel_used = True
        except Exception as e:
            print(f"  Sub-pixel refinement failed: {e}, using integer positions")
            # Fall back to integer positions
            left_edges = edge_data["left_edges"]
            right_edges = edge_data["right_edges"]

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
    
    # DEBUG: Print raw measurements
    print(f"  [DEBUG] Raw median width: {median_width_px:.2f}px, scale: {scale_px_per_cm:.2f} px/cm → {median_width_cm:.4f}cm")
    print(f"  [DEBUG] Width range: {np.min(widths_filtered):.1f}-{np.max(widths_filtered):.1f}px, std: {std_width_px:.1f}px")

    return {
        "widths_px": widths_filtered.tolist(),
        "median_width_px": median_width_px,
        "median_width_cm": median_width_cm,
        "mean_width_px": mean_width_px,
        "std_width_px": std_width_px,
        "num_samples": len(widths_filtered),
        "outliers_removed": int(outliers_removed),
        "subpixel_refinement_used": subpixel_used,
    }


def compute_edge_quality_score(
    gradient_data: Dict[str, Any],
    edge_data: Dict[str, Any],
    width_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Assess quality of edge detection for confidence scoring.

    Computes 4 quality metrics:
    1. Gradient strength: Average gradient magnitude at detected edges
    2. Edge consistency: Percentage of rows with valid edge pairs
    3. Edge smoothness: Variance of edge positions along finger
    4. Bilateral symmetry: Correlation between left/right edge quality

    Args:
        gradient_data: Output from apply_sobel_filters()
        edge_data: Output from detect_edges_per_row()
        width_data: Output from measure_width_from_edges()

    Returns:
        Dictionary containing:
        - overall_score: Weighted average (0-1)
        - gradient_strength_score: Gradient strength metric (0-1)
        - consistency_score: Edge detection success rate (0-1)
        - smoothness_score: Edge position smoothness (0-1)
        - symmetry_score: Left/right balance (0-1)
        - metrics: Dict with raw metric values
    """
    gradient_magnitude = gradient_data["gradient_magnitude"]
    left_edges = edge_data["left_edges"]
    right_edges = edge_data["right_edges"]
    valid_rows = edge_data["valid_rows"]
    edge_strengths_left = edge_data["edge_strengths_left"]
    edge_strengths_right = edge_data["edge_strengths_right"]

    # Metric 1: Gradient Strength
    # Average gradient magnitude at detected edges, normalized
    valid_left_strengths = edge_strengths_left[valid_rows]
    valid_right_strengths = edge_strengths_right[valid_rows]

    if len(valid_left_strengths) > 0:
        avg_gradient_strength = (np.mean(valid_left_strengths) + np.mean(valid_right_strengths)) / 2.0
        # Normalize: typical strong edge is 20-50, weak is <10
        # Score = min(strength / 30, 1.0)
        gradient_strength_score = min(avg_gradient_strength / 30.0, 1.0)
    else:
        avg_gradient_strength = 0.0
        gradient_strength_score = 0.0

    # Metric 2: Edge Consistency
    # Percentage of rows with valid edge pairs
    total_rows = len(valid_rows)
    num_valid = np.sum(valid_rows)
    consistency_score = num_valid / total_rows if total_rows > 0 else 0.0

    # Metric 3: Edge Smoothness
    # Measure variance of edge positions (smoother = better)
    # Lower variance = higher score
    if num_valid > 1:
        # Calculate variance of left and right edges separately
        valid_left = left_edges[valid_rows]
        valid_right = right_edges[valid_rows]

        left_variance = np.var(valid_left)
        right_variance = np.var(valid_right)
        avg_variance = (left_variance + right_variance) / 2.0

        # Normalize: typical finger has variance <100, noisy edges >500
        # Score = exp(-variance / 200) to map variance to 0-1
        smoothness_score = np.exp(-avg_variance / 200.0)
    else:
        avg_variance = 0.0
        smoothness_score = 0.0

    # Metric 4: Bilateral Symmetry
    # Correlation between left and right edge quality (strength balance)
    if len(valid_left_strengths) > 1:
        # Calculate ratio of average strengths
        avg_left = np.mean(valid_left_strengths)
        avg_right = np.mean(valid_right_strengths)

        if avg_left > 0 and avg_right > 0:
            # Symmetric ratio close to 1.0 is good
            ratio = min(avg_left, avg_right) / max(avg_left, avg_right)
            symmetry_score = ratio  # Already 0-1
        else:
            symmetry_score = 0.0
    else:
        symmetry_score = 0.0

    # Weighted overall score
    # Gradient strength: 40% (most important - indicates clear edges)
    # Consistency: 30% (good coverage)
    # Smoothness: 20% (stable detection)
    # Symmetry: 10% (balanced detection)
    overall_score = (
        0.4 * gradient_strength_score +
        0.3 * consistency_score +
        0.2 * smoothness_score +
        0.1 * symmetry_score
    )

    return {
        "overall_score": float(overall_score),
        "gradient_strength_score": float(gradient_strength_score),
        "consistency_score": float(consistency_score),
        "smoothness_score": float(smoothness_score),
        "symmetry_score": float(symmetry_score),
        "metrics": {
            "avg_gradient_strength": float(avg_gradient_strength),
            "edge_consistency_pct": float(consistency_score * 100),
            "avg_variance": float(avg_variance) if num_valid > 1 else 0.0,
            "left_right_strength_ratio": float(symmetry_score),
        }
    }


def should_use_sobel_measurement(
    sobel_result: Dict[str, Any],
    contour_result: Optional[Dict[str, Any]] = None,
    min_quality_score: float = 0.7,
    min_consistency: float = 0.5,
    max_difference_pct: float = 50.0
) -> Tuple[bool, str]:
    """
    Decide whether to use Sobel measurement or fall back to contour.

    Decision criteria:
    1. Edge quality score > min_quality_score (default 0.7)
    2. Edge consistency > min_consistency (default 0.5 = 50%)
    3. If contour available: Sobel and contour agree within max_difference_pct

    Args:
        sobel_result: Output from refine_edges_sobel()
        contour_result: Optional output from compute_cross_section_width()
        min_quality_score: Minimum acceptable quality score
        min_consistency: Minimum edge detection success rate
        max_difference_pct: Maximum allowed difference from contour (%)

    Returns:
        Tuple of (should_use_sobel, reason)
    """
    # Check if edge quality data available
    if "edge_quality" not in sobel_result:
        return False, "edge_quality_data_missing"

    edge_quality = sobel_result["edge_quality"]

    # Check 1: Overall quality score
    if edge_quality["overall_score"] < min_quality_score:
        return False, f"quality_score_low_{edge_quality['overall_score']:.2f}"

    # Check 2: Consistency (success rate)
    if edge_quality["consistency_score"] < min_consistency:
        return False, f"consistency_low_{edge_quality['consistency_score']:.2f}"

    # Check 3: Measurement reasonableness
    sobel_width = sobel_result.get("median_width_cm")
    if sobel_width is None or sobel_width <= 0:
        return False, "invalid_measurement"

    # Typical finger width is 1.0-3.0 cm
    if sobel_width < 0.8 or sobel_width > 3.5:
        return False, f"unrealistic_width_{sobel_width:.2f}cm"

    # Check 4: Agreement with contour (if available)
    if contour_result is not None:
        contour_width = contour_result.get("median_width_px")
        sobel_width_px = sobel_result.get("median_width_px")

        if contour_width and sobel_width_px:
            diff_pct = abs(sobel_width_px - contour_width) / contour_width * 100

            if diff_pct > max_difference_pct:
                return False, f"disagrees_with_contour_{diff_pct:.1f}pct"

    # All checks passed
    return True, "quality_acceptable"


def refine_edges_sobel(
    image: np.ndarray,
    axis_data: Dict[str, Any],
    zone_data: Dict[str, Any],
    scale_px_per_cm: float,
    finger_mask: Optional[np.ndarray] = None,
    finger_landmarks: Optional[np.ndarray] = None,
    sobel_threshold: float = 15.0,
    kernel_size: int = 3,
    rotate_align: bool = False,
    expected_width_px: Optional[float] = None,
    debug_dir: Optional[str] = None,
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
        finger_mask: Optional finger mask for constrained detection
        finger_landmarks: Optional 4x2 array of finger landmarks for debug
        sobel_threshold: Minimum gradient magnitude for valid edge
        kernel_size: Sobel kernel size (3, 5, or 7)
        rotate_align: Rotate ROI for vertical finger alignment
        expected_width_px: Expected width for validation (optional)
        debug_dir: Directory to save debug visualizations (None to skip)

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
    # Initialize debug observer if debug_dir provided
    if debug_dir:
        from src.debug_observer import DebugObserver, draw_landmark_axis, draw_ring_zone_roi
        from src.debug_observer import draw_roi_extraction, draw_gradient_visualization
        from src.debug_observer import draw_edge_candidates, draw_selected_edges
        from src.debug_observer import draw_width_measurements, draw_outlier_detection
        from src.debug_observer import draw_comprehensive_edge_overlay
        observer = DebugObserver(debug_dir)
    
    # Stage A: Axis & Zone Visualization
    if debug_dir:
        # A.1: Landmark axis
        observer.draw_and_save("01_landmark_axis", image, draw_landmark_axis, axis_data, finger_landmarks)
        
        # A.2: Ring zone + ROI bounds (need to extract bounds first)
        # We'll save this after ROI extraction
    
    # Step 1: Extract ROI
    roi_data = extract_ring_zone_roi(
        image, axis_data, zone_data,
        finger_mask=finger_mask,
        padding=50, rotate_align=rotate_align
    )
    
    print(f"  [DEBUG] ROI size: {roi_data['roi_width']}x{roi_data['roi_height']}px")
    print(f"  [DEBUG] ROI bounds: {roi_data['roi_bounds']}")
    
    if debug_dir:
        # A.2: Ring zone + ROI bounds
        roi_bounds = roi_data["roi_bounds"]
        observer.draw_and_save("02_ring_zone_roi", image, draw_ring_zone_roi, zone_data, roi_bounds)
        
        # A.3: ROI extraction
        observer.draw_and_save("03_roi_extraction", roi_data["roi_image"], draw_roi_extraction, roi_data.get("roi_mask"))

    # Step 2: Apply Sobel filters
    gradient_data = apply_sobel_filters(
        roi_data["roi_image"],
        kernel_size=kernel_size,
        axis_direction="auto"
    )
    
    if debug_dir:
        # Stage B: Sobel Filtering
        # B.1: Left-to-right gradient
        grad_left = draw_gradient_visualization(gradient_data["gradient_x"], cv2.COLORMAP_JET)
        observer.save_stage("04_sobel_left_to_right", grad_left)
        
        # B.2: Right-to-left gradient
        grad_right = draw_gradient_visualization(-gradient_data["gradient_x"], cv2.COLORMAP_JET)
        observer.save_stage("05_sobel_right_to_left", grad_right)
        
        # B.3: Gradient magnitude
        grad_mag = draw_gradient_visualization(gradient_data["gradient_magnitude"], cv2.COLORMAP_HOT)
        observer.save_stage("06_gradient_magnitude", grad_mag)

    # Step 3: Detect edges per row
    edge_data = detect_edges_per_row(
        gradient_data, roi_data,
        threshold=sobel_threshold,
        expected_width_px=expected_width_px
    )
    
    print(f"  [DEBUG] Valid rows: {edge_data['num_valid_rows']}/{len(edge_data['valid_rows'])} ({edge_data['num_valid_rows']/len(edge_data['valid_rows'])*100:.1f}%)")
    if edge_data['num_valid_rows'] > 0:
        valid_left = edge_data['left_edges'][edge_data['valid_rows']]
        valid_right = edge_data['right_edges'][edge_data['valid_rows']]
        print(f"  [DEBUG] Left edges range: {np.min(valid_left):.1f}-{np.max(valid_left):.1f}px")
        print(f"  [DEBUG] Right edges range: {np.min(valid_right):.1f}-{np.max(valid_right):.1f}px")
        widths = valid_right - valid_left
        print(f"  [DEBUG] Raw widths range: {np.min(widths):.1f}-{np.max(widths):.1f}px, median: {np.median(widths):.1f}px")
    
    if debug_dir:
        # B.4: Edge candidates
        observer.draw_and_save("07_edge_candidates", roi_data["roi_image"], 
                             draw_edge_candidates, gradient_data["gradient_magnitude"], sobel_threshold)
        
        # B.5: Selected edges
        observer.draw_and_save("08_selected_edges", roi_data["roi_image"], draw_selected_edges, edge_data)

    # Step 4: Measure width from edges (with sub-pixel refinement)
    width_data = measure_width_from_edges(
        edge_data, roi_data, scale_px_per_cm,
        gradient_data=gradient_data,
        use_subpixel=True
    )
    
    if debug_dir:
        # Stage C: Measurement
        # C.1: Sub-pixel refinement (use selected edges for now)
        observer.draw_and_save("09_subpixel_refinement", roi_data["roi_image"], draw_selected_edges, edge_data)
        
        # C.2: Width measurements
        observer.draw_and_save("10_width_measurements", roi_data["roi_image"], 
                             draw_width_measurements, edge_data, width_data)
        
        # C.3: Width distribution (histogram - requires matplotlib)
        try:
            _save_width_distribution(width_data, debug_dir)
        except:
            pass  # Skip if matplotlib not available
        
        # C.4: Outlier detection
        observer.draw_and_save("12_outlier_detection", roi_data["roi_image"], 
                             draw_outlier_detection, edge_data, width_data)
        
        # C.5: Comprehensive edge overlay on full image
        observer.draw_and_save("13_comprehensive_overlay", image, 
                             draw_comprehensive_edge_overlay,
                             edge_data, roi_data["roi_bounds"], axis_data, zone_data,
                             width_data, scale_px_per_cm)

    # Step 5: Compute edge quality score
    edge_quality = compute_edge_quality_score(
        gradient_data, edge_data, width_data
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
        "subpixel_refinement_used": width_data["subpixel_refinement_used"],
        "edge_detection_success_rate": success_rate,
        "edge_quality": edge_quality,
        "roi_data": roi_data,
        "gradient_data": gradient_data,
        "edge_data": edge_data,
        "width_data": width_data,
        "method": "sobel",
    }


def _save_width_distribution(width_data: Dict[str, Any], debug_dir: str) -> None:
    """Helper to save width distribution histogram."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os
    except ImportError:
        return
    
    widths_px = width_data.get("widths_px", [])
    if len(widths_px) == 0:
        return
    
    median_width_px = width_data["median_width_px"]
    mean_width_px = width_data["mean_width_px"]
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(widths_px, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(median_width_px, color='red', linestyle='--', linewidth=2, label=f'Median: {median_width_px:.1f} px')
    ax.axvline(mean_width_px, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean_width_px:.1f} px')
    
    ax.set_xlabel('Width (pixels)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Cross-Section Widths', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save
    output_path = os.path.join(debug_dir, "11_width_distribution.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compare_edge_methods(
    contour_result: Dict[str, Any],
    sobel_result: Dict[str, Any],
    scale_px_per_cm: float
) -> Dict[str, Any]:
    """
    Compare contour-based and Sobel-based edge detection methods.

    Provides detailed analysis of differences, quality metrics, and
    recommendation on which method to use.

    Args:
        contour_result: Output from compute_cross_section_width()
        sobel_result: Output from refine_edges_sobel()
        scale_px_per_cm: Scale factor for unit conversion

    Returns:
        Dictionary containing:
        - contour: Summary of contour method results
        - sobel: Summary of Sobel method results
        - difference: Comparison metrics
        - recommendation: Which method to use and why
        - quality_comparison: Quality metrics comparison
    """
    # Extract measurements
    contour_width_cm = contour_result["median_width_px"] / scale_px_per_cm
    sobel_width_cm = sobel_result["median_width_cm"]

    contour_width_px = contour_result["median_width_px"]
    sobel_width_px = sobel_result["median_width_px"]

    # Calculate differences
    diff_cm = sobel_width_cm - contour_width_cm
    diff_px = sobel_width_px - contour_width_px
    diff_pct = (diff_cm / contour_width_cm) * 100 if contour_width_cm > 0 else 0.0

    # Quality comparison
    contour_cv = (contour_result["std_width_px"] / contour_result["median_width_px"]) if contour_result["median_width_px"] > 0 else 0.0
    sobel_cv = (sobel_result["std_width_px"] / sobel_result["median_width_px"]) if sobel_result["median_width_px"] > 0 else 0.0

    # Determine recommendation
    should_use_sobel, reason = should_use_sobel_measurement(sobel_result, contour_result)

    # Build summary
    result = {
        "contour": {
            "width_cm": float(contour_width_cm),
            "width_px": float(contour_width_px),
            "std_dev_px": float(contour_result["std_width_px"]),
            "coefficient_variation": float(contour_cv),
            "num_samples": int(contour_result["num_samples"]),
            "method": "contour",
        },
        "sobel": {
            "width_cm": float(sobel_width_cm),
            "width_px": float(sobel_width_px),
            "std_dev_px": float(sobel_result["std_width_px"]),
            "coefficient_variation": float(sobel_cv),
            "num_samples": int(sobel_result["num_samples"]),
            "subpixel_used": bool(sobel_result["subpixel_refinement_used"]),
            "success_rate": float(sobel_result["edge_detection_success_rate"]),
            "edge_quality_score": float(sobel_result["edge_quality"]["overall_score"]),
            "method": "sobel",
        },
        "difference": {
            "absolute_cm": float(diff_cm),
            "absolute_px": float(diff_px),
            "relative_pct": float(diff_pct),
            "precision_improvement": float(contour_result["std_width_px"] - sobel_result["std_width_px"]),
        },
        "recommendation": {
            "use_sobel": bool(should_use_sobel),
            "reason": str(reason),
            "preferred_method": "sobel" if should_use_sobel else "contour",
        },
        "quality_comparison": {
            "contour_cv": float(contour_cv),
            "sobel_cv": float(sobel_cv),
            "sobel_quality_score": float(sobel_result["edge_quality"]["overall_score"]),
            "sobel_gradient_strength": float(sobel_result["edge_quality"]["gradient_strength_score"]),
            "sobel_consistency": float(sobel_result["edge_quality"]["consistency_score"]),
            "sobel_smoothness": float(sobel_result["edge_quality"]["smoothness_score"]),
            "sobel_symmetry": float(sobel_result["edge_quality"]["symmetry_score"]),
        },
    }

    return result
