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
from typing import Optional, Tuple, Dict, Any, List

# Standard credit card dimensions (ISO/IEC 7810 ID-1)
CARD_WIDTH_MM = 85.60
CARD_HEIGHT_MM = 53.98
CARD_WIDTH_CM = CARD_WIDTH_MM / 10
CARD_HEIGHT_CM = CARD_HEIGHT_MM / 10
CARD_ASPECT_RATIO = CARD_WIDTH_MM / CARD_HEIGHT_MM  # ~1.586

# Detection parameters
MIN_CARD_AREA_RATIO = 0.01  # Card must be at least 1% of image area
MAX_CARD_AREA_RATIO = 0.5   # Card must be at most 50% of image area
CORNER_ANGLE_TOLERANCE = 25  # Degrees deviation from 90° allowed


def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    Order corners as: top-left, top-right, bottom-right, bottom-left.

    Args:
        corners: 4x2 array of corner points

    Returns:
        Ordered 4x2 array of corners
    """
    corners = corners.reshape(4, 2).astype(np.float32)

    # Sort by sum (x+y): smallest = top-left, largest = bottom-right
    s = corners.sum(axis=1)
    tl_idx = np.argmin(s)
    br_idx = np.argmax(s)

    # Sort by diff (y-x): smallest = top-right, largest = bottom-left
    d = np.diff(corners, axis=1).flatten()
    tr_idx = np.argmin(d)
    bl_idx = np.argmax(d)

    return np.array([
        corners[tl_idx],
        corners[tr_idx],
        corners[br_idx],
        corners[bl_idx],
    ], dtype=np.float32)


def compute_corner_angles(corners: np.ndarray) -> List[float]:
    """
    Compute interior angles at each corner of a quadrilateral.

    Args:
        corners: Ordered 4x2 array of corner points

    Returns:
        List of 4 angles in degrees
    """
    angles = []
    n = len(corners)
    for i in range(n):
        p1 = corners[(i - 1) % n]
        p2 = corners[i]
        p3 = corners[(i + 1) % n]

        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        angles.append(angle)

    return angles


def get_quad_dimensions(corners: np.ndarray) -> Tuple[float, float]:
    """
    Get width and height of a quadrilateral from ordered corners.

    Args:
        corners: Ordered 4x2 array (TL, TR, BR, BL)

    Returns:
        Tuple of (width, height) in pixels
    """
    # Width: average of top and bottom edges
    top_width = np.linalg.norm(corners[1] - corners[0])
    bottom_width = np.linalg.norm(corners[2] - corners[3])
    width = (top_width + bottom_width) / 2

    # Height: average of left and right edges
    left_height = np.linalg.norm(corners[3] - corners[0])
    right_height = np.linalg.norm(corners[2] - corners[1])
    height = (left_height + right_height) / 2

    return width, height


def score_card_candidate(
    contour: np.ndarray,
    corners: np.ndarray,
    image_area: float,
    aspect_ratio_tolerance: float = 0.15,
) -> Tuple[float, Dict[str, Any]]:
    """
    Score a quadrilateral candidate for being a credit card.

    Args:
        contour: Original contour
        corners: 4 corner points
        image_area: Total image area for relative sizing
        aspect_ratio_tolerance: Allowed deviation from standard ratio

    Returns:
        Tuple of (score, details_dict)
    """
    ordered = order_corners(corners)
    width, height = get_quad_dimensions(ordered)
    area = cv2.contourArea(corners)

    details = {
        "corners": ordered,
        "width": width,
        "height": height,
        "area": area,
    }

    # Check area ratio
    area_ratio = area / image_area
    if area_ratio < MIN_CARD_AREA_RATIO or area_ratio > MAX_CARD_AREA_RATIO:
        details["reject_reason"] = f"area_ratio={area_ratio:.3f}"
        return 0.0, details

    # Calculate aspect ratio (always use larger/smaller for consistency)
    if width > height:
        aspect_ratio = width / height
    else:
        aspect_ratio = height / width
    details["aspect_ratio"] = aspect_ratio

    # Check aspect ratio against credit card standard
    ratio_diff = abs(aspect_ratio - CARD_ASPECT_RATIO) / CARD_ASPECT_RATIO
    if ratio_diff > aspect_ratio_tolerance:
        details["reject_reason"] = f"aspect_ratio={aspect_ratio:.3f}, expected~{CARD_ASPECT_RATIO:.3f}"
        return 0.0, details

    # Check corner angles (should be close to 90°)
    angles = compute_corner_angles(ordered)
    details["angles"] = angles
    angle_deviations = [abs(a - 90) for a in angles]
    max_deviation = max(angle_deviations)
    if max_deviation > CORNER_ANGLE_TOLERANCE:
        details["reject_reason"] = f"corner_angle_deviation={max_deviation:.1f}°"
        return 0.0, details

    # Check convexity
    if not cv2.isContourConvex(ordered.astype(np.int32)):
        details["reject_reason"] = "not_convex"
        return 0.0, details

    # Compute score (higher is better)
    # Favor: larger area, closer aspect ratio, more rectangular angles
    area_score = min(area_ratio / 0.1, 1.0)  # Normalize to max at 10% of image
    ratio_score = 1.0 - ratio_diff / aspect_ratio_tolerance
    angle_score = 1.0 - max_deviation / CORNER_ANGLE_TOLERANCE

    score = 0.4 * area_score + 0.3 * ratio_score + 0.3 * angle_score
    details["score_components"] = {
        "area": area_score,
        "ratio": ratio_score,
        "angle": angle_score,
    }

    return score, details


def find_card_contours(image: np.ndarray) -> List[np.ndarray]:
    """
    Find potential card contours using multiple detection strategies.

    Args:
        image: Input BGR image

    Returns:
        List of 4-point contour approximations
    """
    h, w = image.shape[:2]
    min_area = h * w * 0.01  # At least 1% of image
    max_area = h * w * 0.5   # At most 50% of image

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges
    filtered = cv2.bilateralFilter(gray, 11, 75, 75)

    candidates = []

    def extract_quads(contours, epsilon_factor=0.02):
        """Extract quadrilaterals from contours with different approximation factors."""
        quads = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            peri = cv2.arcLength(contour, True)

            # Try multiple epsilon values for polygon approximation
            for eps in [epsilon_factor, epsilon_factor * 1.5, epsilon_factor * 2, epsilon_factor * 0.5]:
                approx = cv2.approxPolyDP(contour, eps * peri, True)

                if len(approx) == 4:
                    quads.append(approx)
                    break
                elif len(approx) > 4 and len(approx) <= 8:
                    # Try to find 4 dominant corners using convex hull
                    hull = cv2.convexHull(approx)
                    if len(hull) >= 4:
                        # Get the 4 corners with maximum area
                        hull_approx = cv2.approxPolyDP(hull, 0.05 * cv2.arcLength(hull, True), True)
                        if len(hull_approx) == 4:
                            quads.append(hull_approx)
                            break

        return quads

    # Strategy 1: Canny edge detection with various thresholds
    for canny_low, canny_high in [(20, 60), (30, 100), (50, 150), (75, 200), (100, 250)]:
        edges = cv2.Canny(filtered, canny_low, canny_high)

        # Morphological closing to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find all contours (not just external)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates.extend(extract_quads(contours))

    # Strategy 2: Adaptive thresholding (for varying lighting)
    for block_size in [11, 21, 31, 51]:
        for C in [2, 5, 10]:
            thresh = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, C
            )
            # Invert if needed
            for img in [thresh, 255 - thresh]:
                contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                candidates.extend(extract_quads(contours))

    # Strategy 3: Otsu's thresholding
    _, otsu = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for img in [otsu, 255 - otsu]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates.extend(extract_quads(contours))

    # Strategy 4: Color-based segmentation (gray card on light background)
    # Look for regions that are darker than the background
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]  # Saturation channel

    # Low saturation = gray colors (like metallic card)
    _, low_sat_mask = cv2.threshold(sat, 30, 255, cv2.THRESH_BINARY_INV)

    # Combine with value channel to find gray regions
    val = hsv[:, :, 2]
    gray_mask = cv2.bitwise_and(low_sat_mask, cv2.inRange(val, 80, 200))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.extend(extract_quads(contours, epsilon_factor=0.03))

    return candidates


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
        - width_px, height_px: Detected dimensions
        - aspect_ratio: Detected aspect ratio
        Or None if no card detected
    """
    h, w = image.shape[:2]
    image_area = h * w

    # Find candidate contours
    candidates = find_card_contours(image)

    if not candidates:
        return None

    # Score each candidate
    best_score = 0.0
    best_result = None

    for contour in candidates:
        corners = contour.reshape(4, 2)
        score, details = score_card_candidate(
            contour, corners, image_area, aspect_ratio_tolerance
        )

        if score > best_score:
            best_score = score
            best_result = details

    if best_result is None or best_score < 0.3:
        return None

    return {
        "corners": best_result["corners"],
        "contour": best_result["corners"],
        "confidence": best_score,
        "width_px": best_result["width"],
        "height_px": best_result["height"],
        "aspect_ratio": best_result["aspect_ratio"],
    }


def rectify_card(
    image: np.ndarray,
    corners: np.ndarray,
    output_width: int = 856,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply perspective transform to rectify the card region.

    Args:
        image: Input BGR image
        corners: Ordered 4x2 array of corner points (TL, TR, BR, BL)
        output_width: Width of output image (height computed from aspect ratio)

    Returns:
        Tuple of (rectified_image, transform_matrix)
    """
    corners = corners.astype(np.float32)

    # Determine if card is in portrait or landscape orientation
    width, height = get_quad_dimensions(corners)

    if width > height:
        # Landscape orientation
        out_w = output_width
        out_h = int(output_width / CARD_ASPECT_RATIO)
    else:
        # Portrait orientation (rotated 90°)
        out_h = output_width
        out_w = int(output_width / CARD_ASPECT_RATIO)

    # Destination points
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    # Compute perspective transform
    M = cv2.getPerspectiveTransform(corners, dst)

    # Apply transform
    rectified = cv2.warpPerspective(image, M, (out_w, out_h))

    return rectified, M


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
    width_px, height_px = get_quad_dimensions(corners)

    # Determine orientation and compute scale
    if width_px > height_px:
        # Landscape: width corresponds to card width (8.56 cm)
        px_per_cm_w = width_px / CARD_WIDTH_CM
        px_per_cm_h = height_px / CARD_HEIGHT_CM
    else:
        # Portrait: width corresponds to card height (5.398 cm)
        px_per_cm_w = width_px / CARD_HEIGHT_CM
        px_per_cm_h = height_px / CARD_WIDTH_CM

    # Average the two estimates
    px_per_cm = (px_per_cm_w + px_per_cm_h) / 2

    # Confidence based on consistency between width and height estimates
    consistency = 1.0 - abs(px_per_cm_w - px_per_cm_h) / max(px_per_cm_w, px_per_cm_h)
    confidence = max(0.0, min(1.0, consistency))

    return px_per_cm, confidence
