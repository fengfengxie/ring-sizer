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
    # TODO: Implement in Phase 5
    raise NotImplementedError("Axis estimation will be implemented in Phase 5")


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
    # TODO: Implement in Phase 6
    raise NotImplementedError("Zone localization will be implemented in Phase 6")


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
