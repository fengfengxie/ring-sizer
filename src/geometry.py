"""
Geometric computation utilities.

This module handles:
- Finger axis estimation (PCA)
- Ring-wearing zone localization
- Cross-section width measurement
- Coordinate transformations
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any


def estimate_finger_axis(
    mask: np.ndarray,
    landmarks: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Estimate the principal axis of a finger using PCA.

    Args:
        mask: Binary finger mask
        landmarks: Optional finger landmarks for orientation

    Returns:
        Dictionary containing:
        - center: Axis center point (x, y)
        - direction: Unit direction vector (dx, dy)
        - length: Estimated finger length in pixels
        - palm_end: Palm-side endpoint
        - tip_end: Fingertip endpoint
    """
    # Get all non-zero points in the mask
    points = np.column_stack(np.where(mask > 0))  # Returns (row, col) i.e., (y, x)
    points = points[:, [1, 0]]  # Convert to (x, y) format

    if len(points) < 10:
        raise ValueError("Not enough points in mask for axis estimation")

    # Calculate center (centroid)
    center = np.mean(points, axis=0)

    # Center the points
    centered = points - center

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Principal axis is the eigenvector with largest eigenvalue
    principal_idx = np.argmax(eigenvalues)
    direction = eigenvectors[:, principal_idx]

    # Ensure direction is a unit vector
    direction = direction / np.linalg.norm(direction)

    # Project all points onto the principal axis to find endpoints
    projections = np.dot(centered, direction)
    min_proj = np.min(projections)
    max_proj = np.max(projections)

    # Calculate finger length
    length = max_proj - min_proj

    # Calculate endpoints along the axis
    endpoint1 = center + direction * min_proj
    endpoint2 = center + direction * max_proj

    # Determine which endpoint is palm vs tip
    # If landmarks are provided, use them for orientation
    if landmarks is not None and len(landmarks) == 4:
        # landmarks[0] is MCP (palm side), landmarks[3] is tip
        base_point = landmarks[0]
        tip_point = landmarks[3]

        # Determine which endpoint is closer to the base
        dist1_to_base = np.linalg.norm(endpoint1 - base_point)
        dist2_to_base = np.linalg.norm(endpoint2 - base_point)

        if dist1_to_base < dist2_to_base:
            palm_end = endpoint1
            tip_end = endpoint2
        else:
            palm_end = endpoint2
            tip_end = endpoint1
            direction = -direction  # Flip direction to point from palm to tip
    else:
        # Without landmarks, use heuristic: tip is usually thinner
        # Sample points near each endpoint
        sample_distance = length * 0.1

        # Points near endpoint1
        near_ep1 = points[np.abs(projections - min_proj) < sample_distance]
        # Points near endpoint2
        near_ep2 = points[np.abs(projections - max_proj) < sample_distance]

        # Calculate average distance from axis for each end (proxy for thickness)
        if len(near_ep1) > 0 and len(near_ep2) > 0:
            # Project distances perpendicular to axis
            perp_direction = np.array([-direction[1], direction[0]])
            dist1 = np.mean(np.abs(np.dot(near_ep1 - center, perp_direction)))
            dist2 = np.mean(np.abs(np.dot(near_ep2 - center, perp_direction)))

            # Thinner end is likely the tip
            if dist1 < dist2:
                palm_end = endpoint2
                tip_end = endpoint1
                direction = -direction
            else:
                palm_end = endpoint1
                tip_end = endpoint2
        else:
            # Fallback: assume endpoint2 is tip (positive direction)
            palm_end = endpoint1
            tip_end = endpoint2

    return {
        "center": center.astype(np.float32),
        "direction": direction.astype(np.float32),
        "length": float(length),
        "palm_end": palm_end.astype(np.float32),
        "tip_end": tip_end.astype(np.float32),
    }


def localize_ring_zone(
    axis_data: Dict[str, Any],
    zone_start_pct: float = 0.15,
    zone_end_pct: float = 0.25,
) -> Dict[str, Any]:
    """
    Localize the ring-wearing zone along the finger axis.

    Args:
        axis_data: Output from estimate_finger_axis()
        zone_start_pct: Zone start as percentage from palm (default 15%)
        zone_end_pct: Zone end as percentage from palm (default 25%)

    Returns:
        Dictionary containing:
        - start_point: Zone start position (x, y)
        - end_point: Zone end position (x, y)
        - center_point: Zone center position
        - length: Zone length in pixels
    """
    # Extract axis information
    palm_end = axis_data["palm_end"]
    tip_end = axis_data["tip_end"]
    direction = axis_data["direction"]
    finger_length = axis_data["length"]

    # Calculate zone positions along the axis
    # Start at zone_start_pct from palm end
    start_distance = finger_length * zone_start_pct
    start_point = palm_end + direction * start_distance

    # End at zone_end_pct from palm end
    end_distance = finger_length * zone_end_pct
    end_point = palm_end + direction * end_distance

    # Calculate zone center
    center_point = (start_point + end_point) / 2.0

    # Zone length
    zone_length = end_distance - start_distance

    return {
        "start_point": start_point.astype(np.float32),
        "end_point": end_point.astype(np.float32),
        "center_point": center_point.astype(np.float32),
        "length": float(zone_length),
        "start_pct": zone_start_pct,
        "end_pct": zone_end_pct,
    }


def compute_cross_section_width(
    contour: np.ndarray,
    axis_data: Dict[str, Any],
    zone_data: Dict[str, Any],
    num_samples: int = 20,
) -> Dict[str, Any]:
    """
    Measure finger width by sampling cross-sections perpendicular to axis.

    Args:
        contour: Finger contour points (Nx2)
        axis_data: Output from estimate_finger_axis()
        zone_data: Output from localize_ring_zone()
        num_samples: Number of cross-section samples

    Returns:
        Dictionary containing:
        - widths_px: List of width measurements in pixels
        - sample_points: List of (left, right) intersection points
        - median_width_px: Median width in pixels
        - std_width_px: Standard deviation of widths
        - mean_width_px: Mean width in pixels
    """
    direction = axis_data["direction"]
    start_point = zone_data["start_point"]
    end_point = zone_data["end_point"]

    # Perpendicular direction (rotate 90 degrees)
    perp_direction = np.array([-direction[1], direction[0]], dtype=np.float32)

    widths = []
    sample_points_list = []

    # Generate sample points along the zone
    for i in range(num_samples):
        # Interpolate between start and end
        t = i / (num_samples - 1) if num_samples > 1 else 0.5
        sample_center = start_point + t * (end_point - start_point)

        # Find intersections with contour along perpendicular line
        intersections = line_contour_intersections(
            contour, sample_center, perp_direction
        )

        if len(intersections) >= 2:
            # Convert to numpy array for distance calculations
            pts = np.array(intersections)

            # Find the two points that are farthest apart
            # This handles cases where the line intersects multiple times
            max_dist = 0
            best_pair = None

            for j in range(len(pts)):
                for k in range(j + 1, len(pts)):
                    dist = np.linalg.norm(pts[j] - pts[k])
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = (pts[j], pts[k])

            if best_pair is not None:
                widths.append(max_dist)
                sample_points_list.append(best_pair)

    if len(widths) == 0:
        raise ValueError("No valid width measurements found in ring zone")

    widths = np.array(widths)

    # Calculate statistics
    median_width = float(np.median(widths))
    mean_width = float(np.mean(widths))
    std_width = float(np.std(widths))

    return {
        "widths_px": widths.tolist(),
        "sample_points": sample_points_list,
        "median_width_px": median_width,
        "mean_width_px": mean_width,
        "std_width_px": std_width,
        "num_samples": len(widths),
    }


def line_contour_intersections(
    contour: np.ndarray,
    point: Tuple[float, float],
    direction: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """
    Find intersection points between a line and a contour.

    Args:
        contour: Contour points (Nx2)
        point: A point on the line (x, y)
        direction: Line direction vector (dx, dy)

    Returns:
        List of intersection points
    """
    intersections = []

    # Normalize direction
    direction = np.array(direction, dtype=np.float32)
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    point = np.array(point, dtype=np.float32)

    # Check each edge of the contour
    n = len(contour)
    for i in range(n):
        p1 = contour[i]
        p2 = contour[(i + 1) % n]

        # Find intersection between line and edge segment
        # Line: P = point + t * direction
        # Segment: Q = p1 + s * (p2 - p1), where s ∈ [0, 1]

        edge_vec = p2 - p1

        # Solve: point + t * direction = p1 + s * edge_vec
        # Rearranged: t * direction - s * edge_vec = p1 - point

        # Create matrix [direction, -edge_vec] * [t, s]^T = p1 - point
        A = np.column_stack([direction, -edge_vec])
        b = p1 - point

        # Check if matrix is singular (parallel lines)
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-8:
            continue

        # Solve for t and s
        try:
            params = np.linalg.solve(A, b)
            t, s = params[0], params[1]

            # Check if intersection is on the edge segment (s ∈ [0, 1])
            if 0 <= s <= 1:
                intersection = point + t * direction
                intersections.append(tuple(intersection))
        except np.linalg.LinAlgError:
            continue

    return intersections
