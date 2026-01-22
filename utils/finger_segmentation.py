"""
Hand and finger segmentation utilities.

This module handles:
- Hand detection using MediaPipe
- Hand mask generation
- Individual finger isolation
- Mask cleanup and validation
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any, Literal

FingerIndex = Literal["auto", "index", "middle", "ring", "pinky"]

# MediaPipe hand landmark indices for each finger
FINGER_LANDMARKS = {
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


def segment_hand(
    image: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Detect and segment hand from image using MediaPipe.

    Args:
        image: Input BGR image

    Returns:
        Dictionary containing:
        - landmarks: 21x2 array of landmark positions
        - mask: Binary hand mask
        - confidence: Detection confidence
        Or None if no hand detected
    """
    # TODO: Implement in Phase 4
    raise NotImplementedError("Hand segmentation will be implemented in Phase 4")


def isolate_finger(
    hand_data: Dict[str, Any],
    finger: FingerIndex = "auto",
) -> Optional[Dict[str, Any]]:
    """
    Isolate a specific finger from hand segmentation data.

    Args:
        hand_data: Output from segment_hand()
        finger: Which finger to isolate, or "auto" to select largest extended

    Returns:
        Dictionary containing:
        - mask: Binary finger mask
        - landmarks: Finger landmark positions
        - base_point: Palm-side base of finger
        - tip_point: Fingertip position
        Or None if finger cannot be isolated
    """
    # TODO: Implement in Phase 4
    raise NotImplementedError("Finger isolation will be implemented in Phase 4")


def clean_mask(
    mask: np.ndarray,
    min_area: int = 1000,
) -> Optional[np.ndarray]:
    """
    Clean a binary mask by extracting largest component and applying morphology.

    Args:
        mask: Input binary mask
        min_area: Minimum valid area in pixels

    Returns:
        Cleaned binary mask, or None if no valid component found
    """
    # TODO: Implement in Phase 4
    raise NotImplementedError("Mask cleaning will be implemented in Phase 4")


def get_finger_contour(
    mask: np.ndarray,
    smooth: bool = True,
) -> Optional[np.ndarray]:
    """
    Extract outer contour from finger mask.

    Args:
        mask: Binary finger mask
        smooth: Whether to apply contour smoothing

    Returns:
        Contour points as Nx2 array, or None if no contour found
    """
    # TODO: Implement in Phase 4
    raise NotImplementedError("Contour extraction will be implemented in Phase 4")
