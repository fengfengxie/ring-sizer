"""
Confidence scoring utilities.

This module handles:
- Card detection confidence
- Finger detection confidence
- Measurement stability confidence
- Edge quality confidence (v1)
- Aggregate confidence calculation

All thresholds and weights are imported from confidence_constants.py.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Literal

from .confidence_constants import (
    # Card confidence constants
    CARD_IDEAL_ASPECT_RATIO,
    CARD_MAX_ASPECT_DEVIATION,
    CARD_WEIGHT_DETECTION,
    CARD_WEIGHT_ASPECT,
    CARD_WEIGHT_SCALE,
    # Finger confidence constants
    FINGER_IDEAL_MIN_AREA_FRACTION,
    FINGER_IDEAL_MAX_AREA_FRACTION,
    FINGER_WEIGHT_HAND_DETECTION,
    FINGER_WEIGHT_MASK_VALIDITY,
    # Measurement confidence constants
    MEASUREMENT_CV_POOR,
    MEASUREMENT_CONSISTENCY_THRESHOLD,
    MEASUREMENT_OUTLIER_STD_MULTIPLIER,
    MEASUREMENT_WIDTH_TYPICAL_MIN,
    MEASUREMENT_WIDTH_TYPICAL_MAX,
    MEASUREMENT_WIDTH_ABSOLUTE_MIN,
    MEASUREMENT_WIDTH_ABSOLUTE_MAX,
    MEASUREMENT_WEIGHT_VARIANCE,
    MEASUREMENT_WEIGHT_CONSISTENCY,
    MEASUREMENT_WEIGHT_OUTLIERS,
    MEASUREMENT_WEIGHT_RANGE,
    MEASUREMENT_RANGE_SCORE_IDEAL,
    MEASUREMENT_RANGE_SCORE_BORDERLINE,
    MEASUREMENT_RANGE_SCORE_OUTSIDE,
    # Overall confidence constants
    V0_WEIGHT_CARD,
    V0_WEIGHT_FINGER,
    V0_WEIGHT_MEASUREMENT,
    V1_WEIGHT_CARD,
    V1_WEIGHT_FINGER,
    V1_WEIGHT_EDGE_QUALITY,
    V1_WEIGHT_MEASUREMENT,
    CONFIDENCE_LEVEL_HIGH_THRESHOLD,
    CONFIDENCE_LEVEL_MEDIUM_THRESHOLD,
)

logger = logging.getLogger(__name__)

EdgeMethod = Literal["contour", "sobel", "sobel_fallback"]


def compute_card_confidence(
    card_result: Dict[str, Any],
    scale_confidence: float,
) -> float:
    """
    Compute confidence score from card detection.

    Uses constants:
    - CARD_IDEAL_ASPECT_RATIO: ISO/IEC 7810 ID-1 aspect ratio
    - CARD_MAX_ASPECT_DEVIATION: Maximum acceptable deviation (0.15)
    - CARD_WEIGHT_*: Component weights (detection: 50%, aspect: 25%, scale: 25%)

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
    aspect_deviation = abs(aspect_ratio - CARD_IDEAL_ASPECT_RATIO) / CARD_IDEAL_ASPECT_RATIO

    # Penalize deviation beyond threshold
    aspect_score = max(0, 1.0 - (aspect_deviation / CARD_MAX_ASPECT_DEVIATION))

    # Combine components with weights
    card_conf = (
        CARD_WEIGHT_DETECTION * detection_conf +
        CARD_WEIGHT_ASPECT * aspect_score +
        CARD_WEIGHT_SCALE * scale_confidence
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

    Uses constants:
    - FINGER_IDEAL_MIN_AREA_FRACTION: Minimum ideal mask area (0.5% of image)
    - FINGER_IDEAL_MAX_AREA_FRACTION: Maximum ideal mask area (5% of image)
    - FINGER_WEIGHT_*: Component weights (hand: 70%, mask: 30%)

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
    # Ideal range: FINGER_IDEAL_MIN_AREA_FRACTION to FINGER_IDEAL_MAX_AREA_FRACTION
    if mask_fraction < FINGER_IDEAL_MIN_AREA_FRACTION:
        area_score = mask_fraction / FINGER_IDEAL_MIN_AREA_FRACTION
    elif mask_fraction > FINGER_IDEAL_MAX_AREA_FRACTION:
        area_score = max(0, 1.0 - (mask_fraction - FINGER_IDEAL_MAX_AREA_FRACTION) / FINGER_IDEAL_MAX_AREA_FRACTION)
    else:
        area_score = 1.0

    # Combine components with weights
    finger_conf = FINGER_WEIGHT_HAND_DETECTION * hand_conf + FINGER_WEIGHT_MASK_VALIDITY * area_score

    return float(np.clip(finger_conf, 0, 1))


def compute_measurement_confidence(
    width_data: Dict[str, Any],
    median_width_cm: float,
) -> float:
    """
    Compute confidence score from measurement stability.

    Uses constants:
    - MEASUREMENT_CV_POOR: Coefficient of variation threshold (0.15)
    - MEASUREMENT_CONSISTENCY_THRESHOLD: Median-mean difference threshold (0.1)
    - MEASUREMENT_OUTLIER_STD_MULTIPLIER: Outlier detection threshold (2.0)
    - MEASUREMENT_WIDTH_*: Realistic width ranges (1.0-3.0 cm)
    - MEASUREMENT_WEIGHT_*: Component weights (variance: 40%, consistency: 20%, outliers: 20%, range: 20%)
    - MEASUREMENT_RANGE_SCORE_*: Range score values

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
    # CV < MEASUREMENT_CV_POOR is acceptable
    variance_score = max(0, 1.0 - coefficient_of_variation / MEASUREMENT_CV_POOR)

    # 2. Median-Mean consistency
    median_mean_diff = abs(median_px - mean_px) / (median_px + 1e-8)
    consistency_score = max(0, 1.0 - median_mean_diff / MEASUREMENT_CONSISTENCY_THRESHOLD)

    # 3. Outlier ratio (measurements far from median)
    outlier_threshold = MEASUREMENT_OUTLIER_STD_MULTIPLIER * std_px
    outliers = np.sum(np.abs(widths_px - median_px) > outlier_threshold)
    outlier_ratio = outliers / len(widths_px)
    outlier_score = max(0, 1.0 - outlier_ratio)

    # 4. Realistic range check
    if MEASUREMENT_WIDTH_TYPICAL_MIN <= median_width_cm <= MEASUREMENT_WIDTH_TYPICAL_MAX:
        range_score = MEASUREMENT_RANGE_SCORE_IDEAL
    elif MEASUREMENT_WIDTH_ABSOLUTE_MIN <= median_width_cm <= MEASUREMENT_WIDTH_ABSOLUTE_MAX:
        # Borderline acceptable
        range_score = MEASUREMENT_RANGE_SCORE_BORDERLINE
    else:
        # Outside realistic range
        range_score = MEASUREMENT_RANGE_SCORE_OUTSIDE

    # Combine components with weights
    measurement_conf = (
        MEASUREMENT_WEIGHT_VARIANCE * variance_score +
        MEASUREMENT_WEIGHT_CONSISTENCY * consistency_score +
        MEASUREMENT_WEIGHT_OUTLIERS * outlier_score +
        MEASUREMENT_WEIGHT_RANGE * range_score
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
    - v0 (contour): 3 components with V0_WEIGHT_* constants
    - v1 (sobel): 4 components with V1_WEIGHT_* constants

    Uses constants:
    - V0_WEIGHT_*: v0 component weights (card: 30%, finger: 30%, measurement: 40%)
    - V1_WEIGHT_*: v1 component weights (card: 25%, finger: 25%, edge: 20%, measurement: 30%)
    - CONFIDENCE_LEVEL_*_THRESHOLD: Level thresholds (high: >0.85, medium: >=0.6)

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
        # v1 scoring: 4 components with V1_WEIGHT_* constants
        overall = (
            V1_WEIGHT_CARD * card_confidence +
            V1_WEIGHT_FINGER * finger_confidence +
            V1_WEIGHT_EDGE_QUALITY * edge_quality_confidence +
            V1_WEIGHT_MEASUREMENT * measurement_confidence
        )
        result["edge_quality"] = float(edge_quality_confidence)

    else:
        # v0 scoring: 3 components with V0_WEIGHT_* constants (contour method or sobel fallback)
        overall = (
            V0_WEIGHT_CARD * card_confidence +
            V0_WEIGHT_FINGER * finger_confidence +
            V0_WEIGHT_MEASUREMENT * measurement_confidence
        )
        result["edge_quality"] = None

    overall = float(np.clip(overall, 0, 1))

    # Classify confidence level using threshold constants
    if overall > CONFIDENCE_LEVEL_HIGH_THRESHOLD:
        level = "high"
    elif overall >= CONFIDENCE_LEVEL_MEDIUM_THRESHOLD:
        level = "medium"
    else:
        level = "low"

    result["overall"] = overall
    result["level"] = level

    return result
