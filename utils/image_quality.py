"""
Image quality assessment utilities.

This module handles:
- Blur detection using Laplacian variance
- Exposure/contrast analysis
- Overall quality scoring
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple


# Quality thresholds
BLUR_THRESHOLD = 50.0  # Laplacian variance below this is considered blurry
MIN_BRIGHTNESS = 40  # Mean brightness below this is underexposed
MAX_BRIGHTNESS = 220  # Mean brightness above this is overexposed
MIN_CONTRAST = 30  # Std dev below this indicates low contrast


def detect_blur(image: np.ndarray) -> Tuple[float, bool]:
    """
    Detect image blur using Laplacian variance method.

    The Laplacian operator highlights regions of rapid intensity change,
    so a well-focused image will have high variance in Laplacian response.

    Args:
        image: Input BGR image

    Returns:
        Tuple of (blur_score, is_sharp)
        - blur_score: Laplacian variance (higher = sharper)
        - is_sharp: True if image passes sharpness threshold
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Variance of Laplacian indicates focus quality
    blur_score = laplacian.var()

    is_sharp = blur_score >= BLUR_THRESHOLD

    return blur_score, is_sharp


def check_exposure(image: np.ndarray) -> Dict[str, Any]:
    """
    Check image exposure and contrast using histogram analysis.

    Args:
        image: Input BGR image

    Returns:
        Dictionary containing:
        - brightness: Mean brightness (0-255)
        - contrast: Standard deviation of brightness
        - is_underexposed: True if image is too dark
        - is_overexposed: True if image is too bright
        - has_good_contrast: True if contrast is sufficient
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate statistics
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Check exposure conditions
    is_underexposed = brightness < MIN_BRIGHTNESS
    is_overexposed = brightness > MAX_BRIGHTNESS
    has_good_contrast = contrast >= MIN_CONTRAST

    return {
        "brightness": brightness,
        "contrast": contrast,
        "is_underexposed": is_underexposed,
        "is_overexposed": is_overexposed,
        "has_good_contrast": has_good_contrast,
    }


def check_resolution(image: np.ndarray, min_dimension: int = 720) -> Dict[str, Any]:
    """
    Check if image resolution is sufficient.

    Args:
        image: Input BGR image
        min_dimension: Minimum acceptable dimension (default 720 for 720p)

    Returns:
        Dictionary containing:
        - width: Image width in pixels
        - height: Image height in pixels
        - is_sufficient: True if resolution meets minimum
    """
    height, width = image.shape[:2]
    min_dim = min(width, height)

    return {
        "width": width,
        "height": height,
        "is_sufficient": min_dim >= min_dimension,
    }


def assess_image_quality(image: np.ndarray) -> Dict[str, Any]:
    """
    Comprehensive image quality assessment.

    Combines blur detection, exposure check, and resolution check
    to determine if image is suitable for processing.

    Args:
        image: Input BGR image

    Returns:
        Dictionary containing:
        - passed: True if image passes all quality checks
        - blur_score: Laplacian variance score
        - brightness: Mean brightness
        - contrast: Standard deviation
        - resolution: (width, height)
        - issues: List of quality issues found
        - fail_reason: Primary failure reason if failed, else None
    """
    issues = []
    fail_reason = None

    # Check blur
    blur_score, is_sharp = detect_blur(image)
    if not is_sharp:
        issues.append(f"Image is blurry (score: {blur_score:.1f}, threshold: {BLUR_THRESHOLD})")
        if fail_reason is None:
            fail_reason = "image_too_blurry"

    # Check exposure
    exposure = check_exposure(image)
    if exposure["is_underexposed"]:
        issues.append(f"Image is underexposed (brightness: {exposure['brightness']:.1f})")
        if fail_reason is None:
            fail_reason = "image_underexposed"
    if exposure["is_overexposed"]:
        issues.append(f"Image is overexposed (brightness: {exposure['brightness']:.1f})")
        if fail_reason is None:
            fail_reason = "image_overexposed"
    if not exposure["has_good_contrast"]:
        issues.append(f"Image has low contrast (std: {exposure['contrast']:.1f})")
        if fail_reason is None:
            fail_reason = "image_low_contrast"

    # Check resolution
    resolution = check_resolution(image)
    if not resolution["is_sufficient"]:
        issues.append(
            f"Resolution too low ({resolution['width']}x{resolution['height']})"
        )
        if fail_reason is None:
            fail_reason = "image_resolution_too_low"

    passed = len(issues) == 0

    return {
        "passed": passed,
        "blur_score": round(blur_score, 2),
        "brightness": round(exposure["brightness"], 2),
        "contrast": round(exposure["contrast"], 2),
        "resolution": (resolution["width"], resolution["height"]),
        "issues": issues,
        "fail_reason": fail_reason,
    }
