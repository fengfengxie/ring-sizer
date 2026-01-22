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
    """
    # TODO: Implement in Phase 7
    raise NotImplementedError("Width measurement will be implemented in Phase 7")


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
    # TODO: Implement in Phase 7
    raise NotImplementedError("Line intersection will be implemented in Phase 7")
