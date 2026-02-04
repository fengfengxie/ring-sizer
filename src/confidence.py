"""
Confidence scoring utilities.

This module handles:
- Card detection confidence
- Finger detection confidence
- Measurement stability confidence
- Edge quality confidence (v1)
- Aggregate confidence calculation
"""

import numpy as np
from typing import Dict, Any, Optional, Literal

EdgeMethod = Literal["contour", "sobel", "sobel_fallback"]


def compute_card_confidence(
    card_result: Dict[str, Any],
    scale_confidence: float,
) -> float:
    """
    Compute confidence score from card detection.

    Args:
        card_result: Output from detect_credit_card()
        scale_confidence: Scale calibration confidence

    Returns:
        Card confidence score [0, 1]
    """
    # Base confidence from card detection
    detection_conf = card_result.get("confidence", 0.0)

    # Aspect ratio deviation penalty
    aspect_ratio = card_result.get("aspect_ratio", 0.0)
    ideal_ratio = 85.60 / 53.98  # 1.586
    aspect_deviation = abs(aspect_ratio - ideal_ratio) / ideal_ratio

    # Penalize deviation beyond 5%
    aspect_score = max(0, 1.0 - (aspect_deviation / 0.15))

    # Combine components
    # Detection quality: 50%, Aspect ratio: 25%, Scale calibration: 25%
    card_conf = (
        0.5 * detection_conf +
        0.25 * aspect_score +
        0.25 * scale_confidence
    )

    return float(np.clip(card_conf, 0, 1))


def compute_finger_confidence(
    hand_data: Dict[str, Any],
    finger_data: Dict[str, Any],
    mask_area: int,
    image_area: int,
) -> float:
    """
    Compute confidence score from finger detection.

    Args:
        hand_data: Output from segment_hand()
        finger_data: Output from isolate_finger()
        mask_area: Area of cleaned finger mask in pixels
        image_area: Total image area in pixels

    Returns:
        Finger confidence score [0, 1]
    """
    # Hand landmark detection confidence from MediaPipe
    hand_conf = hand_data.get("confidence", 0.0)

    # Mask area validity (should be reasonable fraction of image)
    mask_fraction = mask_area / image_area
    # Ideal range: 0.5% to 5% of image
    if mask_fraction < 0.005:
        area_score = mask_fraction / 0.005
    elif mask_fraction > 0.05:
        area_score = max(0, 1.0 - (mask_fraction - 0.05) / 0.05)
    else:
        area_score = 1.0

    # Combine components
    # Hand detection: 70%, Mask validity: 30%
    finger_conf = 0.7 * hand_conf + 0.3 * area_score

    return float(np.clip(finger_conf, 0, 1))


def compute_measurement_confidence(
    width_data: Dict[str, Any],
    median_width_cm: float,
) -> float:
    """
    Compute confidence score from measurement stability.

    Args:
        width_data: Output from compute_cross_section_width()
        median_width_cm: Median width in centimeters

    Returns:
        Measurement confidence score [0, 1]
    """
    widths_px = np.array(width_data.get("widths_px", []))

    if len(widths_px) == 0:
        return 0.0

    median_px = width_data.get("median_width_px", 0.0)
    mean_px = width_data.get("mean_width_px", 0.0)
    std_px = width_data.get("std_width_px", 0.0)

    # 1. Variance score (lower variance = higher confidence)
    coefficient_of_variation = std_px / (median_px + 1e-8)
    # CV < 0.05 is excellent, CV > 0.2 is poor
    variance_score = max(0, 1.0 - coefficient_of_variation / 0.15)

    # 2. Median-Mean consistency
    median_mean_diff = abs(median_px - mean_px) / (median_px + 1e-8)
    consistency_score = max(0, 1.0 - median_mean_diff / 0.1)

    # 3. Outlier ratio (measurements far from median)
    outlier_threshold = 2.0 * std_px
    outliers = np.sum(np.abs(widths_px - median_px) > outlier_threshold)
    outlier_ratio = outliers / len(widths_px)
    outlier_score = max(0, 1.0 - outlier_ratio)

    # 4. Realistic range check
    if 1.4 <= median_width_cm <= 2.4:
        range_score = 1.0
    elif 1.0 <= median_width_cm <= 3.0:
        # Borderline acceptable
        range_score = 0.7
    else:
        # Outside realistic range
        range_score = 0.3

    # Combine components
    # Variance: 40%, Consistency: 20%, Outliers: 20%, Range: 20%
    measurement_conf = (
        0.4 * variance_score +
        0.2 * consistency_score +
        0.2 * outlier_score +
        0.2 * range_score
    )

    return float(np.clip(measurement_conf, 0, 1))


def compute_edge_quality_confidence(
    edge_quality_data: Optional[Dict[str, Any]] = None
) -> float:
    """
    Compute confidence score from edge quality (v1 Sobel method).

    Args:
        edge_quality_data: Output from compute_edge_quality_score()
                          None if using contour method (v0)

    Returns:
        Edge quality confidence score [0, 1]
        Returns 1.0 for contour method (not applicable)
    """
    if edge_quality_data is None:
        # Contour method - edge quality not applicable
        return 1.0

    # Use overall edge quality score directly
    # It's already a weighted combination of 4 metrics
    edge_conf = edge_quality_data.get("overall_score", 0.0)

    return float(np.clip(edge_conf, 0, 1))


def compute_overall_confidence(
    card_confidence: float,
    finger_confidence: float,
    measurement_confidence: float,
    edge_method: EdgeMethod = "contour",
    edge_quality_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute overall confidence by combining component scores.

    Supports both v0 (contour) and v1 (Sobel) confidence calculation:
    - v0 (contour): 3 components (card, finger, measurement)
    - v1 (sobel): 4 components (card, finger, edge quality, measurement)

    Args:
        card_confidence: Card detection confidence
        finger_confidence: Finger detection confidence
        measurement_confidence: Measurement stability confidence
        edge_method: Edge detection method used
        edge_quality_confidence: Edge quality confidence (v1 only)

    Returns:
        Dictionary containing:
        - overall: Overall confidence [0, 1]
        - card: Card component score
        - finger: Finger component score
        - measurement: Measurement component score
        - edge_quality: Edge quality score (v1 only, None for v0)
        - level: "high", "medium", or "low"
        - method: Edge method used
    """
    result = {
        "card": float(card_confidence),
        "finger": float(finger_confidence),
        "measurement": float(measurement_confidence),
        "method": edge_method,
    }

    # Calculate overall confidence based on method
    if edge_method == "sobel" and edge_quality_confidence is not None:
        # v1 scoring: 4 components
        # Card: 25%, Finger: 25%, Edge Quality: 20%, Measurement: 30%
        overall = (
            0.25 * card_confidence +
            0.25 * finger_confidence +
            0.20 * edge_quality_confidence +
            0.30 * measurement_confidence
        )
        result["edge_quality"] = float(edge_quality_confidence)

    else:
        # v0 scoring: 3 components (contour method or sobel fallback)
        # Card: 30%, Finger: 30%, Measurement: 40%
        overall = (
            0.30 * card_confidence +
            0.30 * finger_confidence +
            0.40 * measurement_confidence
        )
        result["edge_quality"] = None

    overall = float(np.clip(overall, 0, 1))

    # Classify confidence level
    if overall > 0.85:
        level = "high"
    elif overall >= 0.6:
        level = "medium"
    else:
        level = "low"

    result["overall"] = overall
    result["level"] = level

    return result
