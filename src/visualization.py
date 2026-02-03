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

# Font and visualization constants
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_BASE_SCALE = 1.5  # Base font scale at reference height
FONT_REFERENCE_HEIGHT = 1200  # Reference image height for font scaling
FONT_MIN_SCALE = 1.5  # Minimum font scale regardless of image size

# Thickness constants
BASE_LINE_THICKNESS = 4
BASE_TEXT_THICKNESS = 2
BASE_CONTOUR_THICKNESS = 5

# Color constants (BGR format)
COLOR_CARD = (0, 255, 0)  # Green
COLOR_FINGER = (0, 0, 255)  # Red (actually Magenta in practice)
COLOR_AXIS_PALM = (0, 255, 255)  # Cyan
COLOR_AXIS_TIP = (255, 128, 0)  # Orange
COLOR_AXIS_LINE = (255, 255, 0)  # Yellow
COLOR_RING_ZONE = (0, 255, 255)  # Cyan
COLOR_CROSS_SECTION = (0, 128, 255)  # Orange
COLOR_POINT = (255, 0, 0)  # Blue
COLOR_TEXT_WHITE = (255, 255, 255)  # White
COLOR_TEXT_GREEN = (0, 255, 0)  # Green
COLOR_TEXT_RED = (0, 0, 255)  # Red

# Size constants
CORNER_CIRCLE_RADIUS = 15
ENDPOINT_CIRCLE_RADIUS = 15
INTERSECTION_CIRCLE_RADIUS = 8
TEXT_OFFSET = 25
LABEL_OFFSET = 20

# Layout constants
RESULT_TEXT_Y_START = 60
RESULT_TEXT_LINE_HEIGHT = 55
RESULT_TEXT_X_OFFSET = 40


def get_scaled_font_params(image_height: int) -> Dict[str, float]:
    """
    Calculate font parameters scaled to image dimensions.

    Args:
        image_height: Height of the image in pixels

    Returns:
        Dictionary containing scaled font parameters
    """
    font_scale = max(FONT_MIN_SCALE, image_height / FONT_REFERENCE_HEIGHT)
    scale_factor = font_scale / FONT_BASE_SCALE

    return {
        "font_scale": font_scale,
        "text_thickness": int(BASE_TEXT_THICKNESS * scale_factor),
        "line_thickness": int(BASE_LINE_THICKNESS * scale_factor),
        "contour_thickness": int(BASE_CONTOUR_THICKNESS * scale_factor),
        "corner_radius": int(CORNER_CIRCLE_RADIUS * scale_factor),
        "endpoint_radius": int(ENDPOINT_CIRCLE_RADIUS * scale_factor),
        "intersection_radius": int(INTERSECTION_CIRCLE_RADIUS * scale_factor),
        "text_offset": int(TEXT_OFFSET * scale_factor),
        "label_offset": int(LABEL_OFFSET * scale_factor),
        "line_height": int(RESULT_TEXT_LINE_HEIGHT * scale_factor),
        "y_start": int(RESULT_TEXT_Y_START * scale_factor),
        "x_offset": int(RESULT_TEXT_X_OFFSET * scale_factor),
    }


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
    params = get_scaled_font_params(image.shape[0])

    # Draw quadrilateral
    cv2.polylines(image, [corners], isClosed=True, color=COLOR_CARD,
                  thickness=params["contour_thickness"])

    # Draw corner points with labels
    corner_labels = ["TL", "TR", "BR", "BL"]
    for corner, label in zip(corners, corner_labels):
        cv2.circle(image, tuple(corner), params["corner_radius"], COLOR_CARD, -1)
        cv2.putText(
            image,
            label,
            tuple(corner + np.array([params["label_offset"], -params["label_offset"]])),
            FONT_FACE,
            params["font_scale"],
            COLOR_CARD,
            params["text_thickness"],
        )

    # Add scale annotation
    if scale_px_per_cm is not None:
        center = np.mean(corners, axis=0).astype(np.int32)
        text = f"Card: {scale_px_per_cm:.1f} px/cm"
        cv2.putText(
            image,
            text,
            tuple(center),
            FONT_FACE,
            params["font_scale"] * 1.2,
            COLOR_CARD,
            params["text_thickness"],
        )

    return image


def draw_finger_contour(
    image: np.ndarray,
    contour: np.ndarray,
) -> np.ndarray:
    """Draw finger contour."""
    params = get_scaled_font_params(image.shape[0])
    contour_int = contour.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [contour_int], isClosed=True, color=COLOR_FINGER,
                  thickness=params["contour_thickness"])
    return image


def draw_finger_axis(
    image: np.ndarray,
    axis_data: Dict[str, Any],
) -> np.ndarray:
    """Draw finger axis line."""
    palm_end = axis_data["palm_end"].astype(np.int32)
    tip_end = axis_data["tip_end"].astype(np.int32)
    params = get_scaled_font_params(image.shape[0])

    # Draw axis line
    cv2.line(image, tuple(palm_end), tuple(tip_end), COLOR_AXIS_LINE,
             params["line_thickness"])

    # Mark endpoints
    cv2.circle(image, tuple(palm_end), params["endpoint_radius"], COLOR_AXIS_PALM, -1)
    cv2.circle(image, tuple(tip_end), params["endpoint_radius"], COLOR_AXIS_TIP, -1)

    # Add labels
    cv2.putText(
        image,
        "Palm",
        tuple(palm_end + np.array([params["text_offset"], params["text_offset"]])),
        FONT_FACE,
        params["font_scale"],
        COLOR_AXIS_PALM,
        params["text_thickness"],
    )
    cv2.putText(
        image,
        "Tip",
        tuple(tip_end + np.array([params["text_offset"], params["text_offset"]])),
        FONT_FACE,
        params["font_scale"],
        COLOR_AXIS_TIP,
        params["text_thickness"],
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
    cv2.fillPoly(overlay, [zone_poly], COLOR_RING_ZONE)
    cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)

    # Draw zone boundaries
    params = get_scaled_font_params(image.shape[0])
    cv2.line(
        image,
        tuple(start_left.astype(np.int32)),
        tuple(start_right.astype(np.int32)),
        COLOR_RING_ZONE,
        params["line_thickness"],
    )
    cv2.line(
        image,
        tuple(end_left.astype(np.int32)),
        tuple(end_right.astype(np.int32)),
        COLOR_RING_ZONE,
        params["line_thickness"],
    )

    # Add zone label
    label_offset = int(40 * params["font_scale"] / FONT_BASE_SCALE)
    label_pos = zone_data["center_point"].astype(np.int32) + np.array([band_width + label_offset, 0], dtype=np.int32)
    cv2.putText(
        image,
        "Ring Zone",
        tuple(label_pos),
        FONT_FACE,
        params["font_scale"] * 1.2,
        COLOR_RING_ZONE,
        params["text_thickness"],
    )

    return image


def draw_cross_sections(
    image: np.ndarray,
    width_data: Dict[str, Any],
) -> np.ndarray:
    """Draw cross-section sample lines and intersection points."""
    params = get_scaled_font_params(image.shape[0])
    sample_points = width_data.get("sample_points", [])

    for left, right in sample_points:
        left_int = tuple(np.array(left, dtype=np.int32))
        right_int = tuple(np.array(right, dtype=np.int32))

        # Draw cross-section line
        cv2.line(image, left_int, right_int, COLOR_CROSS_SECTION,
                 max(2, params["line_thickness"] // 2))

        # Draw intersection points
        cv2.circle(image, left_int, params["intersection_radius"], COLOR_POINT, -1)
        cv2.circle(image, right_int, params["intersection_radius"], COLOR_POINT, -1)

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
        level_color = COLOR_TEXT_GREEN
    elif confidence >= 0.6:
        level = "MEDIUM"
        level_color = (0, 255, 255)  # Yellow
    else:
        level = "LOW"
        level_color = COLOR_TEXT_RED

    # Build text lines with JSON information
    text_lines = [
        ("=== MEASUREMENT RESULT ===", COLOR_TEXT_WHITE, False),
        (f"Finger Diameter: {measurement_cm:.2f} cm", COLOR_TEXT_WHITE, False),
        (f"Confidence: {confidence:.3f} ({level})", level_color, True),
        ("", COLOR_TEXT_WHITE, False),  # Empty line
        ("=== QUALITY FLAGS ===", COLOR_TEXT_WHITE, False),
        (f"Card Detected: {'YES' if card_detected else 'NO'}", COLOR_TEXT_GREEN if card_detected else COLOR_TEXT_RED, False),
        (f"Finger Detected: {'YES' if finger_detected else 'NO'}", COLOR_TEXT_GREEN if finger_detected else COLOR_TEXT_RED, False),
        (f"View Angle OK: {'YES' if view_angle_ok else 'NO'}", COLOR_TEXT_GREEN if view_angle_ok else COLOR_TEXT_RED, False),
    ]

    # Add scale information if available
    if scale_px_per_cm is not None:
        text_lines.insert(3, (f"Scale: {scale_px_per_cm:.2f} px/cm", COLOR_TEXT_WHITE, False))

    # Get scaled font parameters
    params = get_scaled_font_params(image.shape[0])

    for i, (text, color, is_bold) in enumerate(text_lines):
        if text:  # Skip empty lines for drawing
            thickness = params["text_thickness"] + 1 if is_bold else params["text_thickness"]
            cv2.putText(
                image,
                text,
                (params["x_offset"], params["y_start"] + i * params["line_height"]),
                FONT_FACE,
                params["font_scale"],
                color,
                thickness,
            )

    return image
