"""
Debug visualization utilities.

This module handles:
- Credit card overlay
- Finger contour and axis visualization
- Ring zone highlighting
- Cross-section measurement display
- Result annotation
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple


def create_debug_visualization(
    image: np.ndarray,
    card_result: Optional[Dict[str, Any]] = None,
    contour: Optional[np.ndarray] = None,
    axis_data: Optional[Dict[str, Any]] = None,
    zone_data: Optional[Dict[str, Any]] = None,
    width_data: Optional[Dict[str, Any]] = None,
    measurement_cm: Optional[float] = None,
    confidence: Optional[float] = None,
    scale_px_per_cm: Optional[float] = None,
) -> np.ndarray:
    """
    Create debug visualization overlay on original image.

    Args:
        image: Original BGR image
        card_result: Credit card detection result
        contour: Finger contour points
        axis_data: Finger axis data
        zone_data: Ring zone data
        width_data: Width measurement data
        measurement_cm: Final measurement in cm
        confidence: Overall confidence score
        scale_px_per_cm: Scale factor

    Returns:
        Annotated BGR image
    """
    # Create a copy for drawing
    vis = image.copy()

    # Draw credit card overlay
    if card_result is not None:
        vis = draw_card_overlay(vis, card_result, scale_px_per_cm)

    # Draw finger contour and axis
    if contour is not None:
        vis = draw_finger_contour(vis, contour)

    if axis_data is not None:
        vis = draw_finger_axis(vis, axis_data)

    # Draw ring zone
    if zone_data is not None and axis_data is not None:
        vis = draw_ring_zone(vis, zone_data, axis_data)

    # Draw cross-section measurements
    if width_data is not None and zone_data is not None:
        vis = draw_cross_sections(vis, width_data)

    # Add measurement annotation with JSON information
    if measurement_cm is not None and confidence is not None:
        vis = add_measurement_text(
            vis,
            measurement_cm,
            confidence,
            scale_px_per_cm=scale_px_per_cm,
            card_detected=card_result is not None,
            finger_detected=contour is not None,
            view_angle_ok=True,  # This is passed from caller
        )

    return vis


def draw_card_overlay(
    image: np.ndarray,
    card_result: Dict[str, Any],
    scale_px_per_cm: Optional[float] = None,
) -> np.ndarray:
    """Draw credit card detection overlay."""
    corners = card_result["corners"].astype(np.int32)

    # Draw quadrilateral
    cv2.polylines(image, [corners], isClosed=True, color=(0, 255, 0), thickness=5)

    # Draw corner points with labels
    corner_labels = ["TL", "TR", "BR", "BL"]
    for i, (corner, label) in enumerate(zip(corners, corner_labels)):
        cv2.circle(image, tuple(corner), 15, (0, 255, 0), -1)
        cv2.putText(
            image,
            label,
            tuple(corner + np.array([20, -20])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            4,
        )

    # Add scale annotation
    if scale_px_per_cm is not None:
        center = np.mean(corners, axis=0).astype(np.int32)
        text = f"Card: {scale_px_per_cm:.1f} px/cm"
        cv2.putText(
            image,
            text,
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (0, 255, 0),
            4,
        )

    return image


def draw_finger_contour(
    image: np.ndarray,
    contour: np.ndarray,
) -> np.ndarray:
    """Draw finger contour."""
    contour_int = contour.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [contour_int], isClosed=True, color=(0, 0, 255), thickness=5)
    return image


def draw_finger_axis(
    image: np.ndarray,
    axis_data: Dict[str, Any],
) -> np.ndarray:
    """Draw finger axis line."""
    palm_end = axis_data["palm_end"].astype(np.int32)
    tip_end = axis_data["tip_end"].astype(np.int32)

    # Draw axis line
    cv2.line(image, tuple(palm_end), tuple(tip_end), (255, 255, 0), 4)

    # Mark endpoints
    cv2.circle(image, tuple(palm_end), 15, (0, 255, 255), -1)
    cv2.circle(image, tuple(tip_end), 15, (255, 128, 0), -1)

    # Add labels
    cv2.putText(
        image,
        "Palm",
        tuple(palm_end + np.array([25, 25])),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 255),
        4,
    )
    cv2.putText(
        image,
        "Tip",
        tuple(tip_end + np.array([25, 25])),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 128, 0),
        4,
    )

    return image


def draw_ring_zone(
    image: np.ndarray,
    zone_data: Dict[str, Any],
    axis_data: Dict[str, Any],
) -> np.ndarray:
    """Draw ring-wearing zone band."""
    direction = axis_data["direction"]
    perp = np.array([-direction[1], direction[0]], dtype=np.float32)

    start_point = zone_data["start_point"]
    end_point = zone_data["end_point"]

    # Create zone band (perpendicular lines at start and end)
    # Make the band wide enough to be visible
    band_width = 200  # pixels

    start_left = start_point + perp * band_width
    start_right = start_point - perp * band_width
    end_left = end_point + perp * band_width
    end_right = end_point - perp * band_width

    # Draw zone band as a semi-transparent overlay
    overlay = image.copy()
    zone_poly = np.array([start_left, start_right, end_right, end_left], dtype=np.int32)
    cv2.fillPoly(overlay, [zone_poly], (0, 255, 255))
    cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)

    # Draw zone boundaries
    cv2.line(
        image,
        tuple(start_left.astype(np.int32)),
        tuple(start_right.astype(np.int32)),
        (0, 255, 255),
        4,
    )
    cv2.line(
        image,
        tuple(end_left.astype(np.int32)),
        tuple(end_right.astype(np.int32)),
        (0, 255, 255),
        4,
    )

    # Add zone label
    label_pos = zone_data["center_point"].astype(np.int32) + np.array([band_width + 40, 0], dtype=np.int32)
    cv2.putText(
        image,
        "Ring Zone",
        tuple(label_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 255, 255),
        4,
    )

    return image


def draw_cross_sections(
    image: np.ndarray,
    width_data: Dict[str, Any],
) -> np.ndarray:
    """Draw cross-section sample lines and intersection points."""
    sample_points = width_data.get("sample_points", [])

    for i, (left, right) in enumerate(sample_points):
        left_int = tuple(np.array(left, dtype=np.int32))
        right_int = tuple(np.array(right, dtype=np.int32))

        # Draw cross-section line
        cv2.line(image, left_int, right_int, (0, 128, 255), 2)

        # Draw intersection points
        cv2.circle(image, left_int, 5, (255, 0, 0), -1)
        cv2.circle(image, right_int, 5, (255, 0, 0), -1)

    return image


def add_measurement_text(
    image: np.ndarray,
    measurement_cm: float,
    confidence: float,
    scale_px_per_cm: Optional[float] = None,
    card_detected: bool = True,
    finger_detected: bool = True,
    view_angle_ok: bool = True,
) -> np.ndarray:
    """Add measurement result text overlay with JSON information."""
    h, w = image.shape[:2]

    # Create larger semi-transparent background for more text
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (1100, 550), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    # Confidence level indicator
    if confidence > 0.85:
        level = "HIGH"
        level_color = (0, 255, 0)
    elif confidence >= 0.6:
        level = "MEDIUM"
        level_color = (0, 255, 255)
    else:
        level = "LOW"
        level_color = (0, 0, 255)

    # Build text lines with JSON information
    text_lines = [
        ("=== MEASUREMENT RESULT ===", (255, 255, 255), False),
        (f"Finger Diameter: {measurement_cm:.2f} cm", (255, 255, 255), False),
        (f"Confidence: {confidence:.3f} ({level})", level_color, True),
        ("", (255, 255, 255), False),  # Empty line
        ("=== QUALITY FLAGS ===", (255, 255, 255), False),
        (f"Card Detected: {'YES' if card_detected else 'NO'}", (0, 255, 0) if card_detected else (0, 0, 255), False),
        (f"Finger Detected: {'YES' if finger_detected else 'NO'}", (0, 255, 0) if finger_detected else (0, 0, 255), False),
        (f"View Angle OK: {'YES' if view_angle_ok else 'NO'}", (0, 255, 0) if view_angle_ok else (0, 0, 255), False),
    ]

    # Add scale information if available
    if scale_px_per_cm is not None:
        text_lines.insert(3, (f"Scale: {scale_px_per_cm:.2f} px/cm", (255, 255, 255), False))

    y_offset = 60
    line_height = 55
    for i, (text, color, is_bold) in enumerate(text_lines):
        if text:  # Skip empty lines for drawing
            thickness = 5 if is_bold else 4
            cv2.putText(
                image,
                text,
                (40, y_offset + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                color,
                thickness,
            )

    return image
