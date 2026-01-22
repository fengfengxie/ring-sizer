"""
Credit card detection and scale calibration utilities.

This module handles:
- Detecting credit card contour in an image
- Verifying aspect ratio matches standard credit card
- Perspective rectification
- Computing pixels-per-cm scale factor
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

# Standard credit card dimensions (ISO/IEC 7810 ID-1)
CARD_WIDTH_MM = 85.60
CARD_HEIGHT_MM = 53.98
CARD_WIDTH_CM = CARD_WIDTH_MM / 10
CARD_HEIGHT_CM = CARD_HEIGHT_MM / 10
CARD_ASPECT_RATIO = CARD_WIDTH_MM / CARD_HEIGHT_MM  # ~1.586


def detect_credit_card(
    image: np.ndarray,
    aspect_ratio_tolerance: float = 0.15,
) -> Optional[Dict[str, Any]]:
    """
    Detect a credit card in the image.

    Args:
        image: Input BGR image
        aspect_ratio_tolerance: Allowed deviation from standard aspect ratio

    Returns:
        Dictionary containing:
        - corners: 4x2 array of corner points (ordered)
        - contour: Full contour points
        - confidence: Detection confidence score
        Or None if no card detected
    """
    # TODO: Implement in Phase 3
    raise NotImplementedError("Card detection will be implemented in Phase 3")


def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    Order corners as: top-left, top-right, bottom-right, bottom-left.

    Args:
        corners: 4x2 array of corner points

    Returns:
        Ordered 4x2 array of corners
    """
    # TODO: Implement in Phase 3
    raise NotImplementedError("Corner ordering will be implemented in Phase 3")


def rectify_card(
    image: np.ndarray,
    corners: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply perspective transform to rectify the card region.

    Args:
        image: Input BGR image
        corners: Ordered 4x2 array of corner points

    Returns:
        Tuple of (rectified_image, transform_matrix)
    """
    # TODO: Implement in Phase 3
    raise NotImplementedError("Card rectification will be implemented in Phase 3")


def compute_scale_factor(
    corners: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute pixels-per-cm scale factor from detected card corners.

    Args:
        corners: Ordered 4x2 array of corner points

    Returns:
        Tuple of (px_per_cm, confidence)
    """
    # TODO: Implement in Phase 3
    raise NotImplementedError("Scale computation will be implemented in Phase 3")
