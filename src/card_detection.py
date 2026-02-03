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
from pathlib import Path

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

# Debug visualization constants
DEBUG_FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
DEBUG_TITLE_FONT_SCALE = 3.5
DEBUG_SUBTITLE_FONT_SCALE = 2.5
DEBUG_LABEL_FONT_SCALE = 1.8
DEBUG_TITLE_THICKNESS = 7
DEBUG_SUBTITLE_THICKNESS = 5
DEBUG_LABEL_THICKNESS = 4
DEBUG_TITLE_OUTLINE_THICKNESS = 10
DEBUG_SUBTITLE_OUTLINE_THICKNESS = 8
DEBUG_LABEL_OUTLINE_THICKNESS = 6
DEBUG_CONTOUR_THICKNESS = 3
DEBUG_CORNER_RADIUS = 8
DEBUG_TITLE_Y = 100
DEBUG_SUBTITLE_Y = 200
DEBUG_LINE_SPACING = 100

# Debug visualization colors (BGR)
DEBUG_COLOR_WHITE = (255, 255, 255)
DEBUG_COLOR_BLACK = (0, 0, 0)
DEBUG_COLOR_GREEN = (0, 255, 0)
DEBUG_COLOR_YELLOW = (0, 255, 255)
DEBUG_COLOR_CYAN = (255, 255, 0)
DEBUG_COLOR_ORANGE = (255, 128, 0)
DEBUG_COLOR_MAGENTA = (255, 0, 255)
DEBUG_COLOR_PINK = (128, 128, 255)


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
    # Safeguard against zero dimensions
    if width <= 0 or height <= 0:
        details["reject_reason"] = "invalid_dimensions"
        return 0.0, details

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


def save_debug_image(image: np.ndarray, filename: str, debug_dir: Optional[str]) -> None:
    """Save a debug image with consistent formatting and compression."""
    if debug_dir is None:
        return

    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(debug_dir) / filename

    # Downsample large images to reduce file size
    h, w = image.shape[:2]
    max_dimension = 1920  # Max width or height
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Use PNG compression for smaller file sizes
    cv2.imwrite(str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def draw_contours_overlay(
    image: np.ndarray,
    contours: List[np.ndarray],
    title: str,
    color: Tuple[int, int, int] = DEBUG_COLOR_GREEN,
) -> np.ndarray:
    """
    Draw contours on an image overlay.

    Args:
        image: Original image
        contours: List of contours to draw
        title: Title for the visualization
        color: BGR color for contours

    Returns:
        Annotated image
    """
    overlay = image.copy()

    # Draw all contours
    for contour in contours:
        if len(contour) == 4:
            # Draw quadrilateral
            pts = contour.reshape(4, 2).astype(np.int32)
            cv2.polylines(overlay, [pts], True, color, DEBUG_CONTOUR_THICKNESS)

    # Add title with outline for visibility
    cv2.putText(
        overlay, title, (20, DEBUG_TITLE_Y),
        DEBUG_FONT_FACE, DEBUG_TITLE_FONT_SCALE, DEBUG_COLOR_WHITE,
        DEBUG_TITLE_OUTLINE_THICKNESS, cv2.LINE_AA
    )
    cv2.putText(
        overlay, title, (20, DEBUG_TITLE_Y),
        DEBUG_FONT_FACE, DEBUG_TITLE_FONT_SCALE, color,
        DEBUG_TITLE_THICKNESS, cv2.LINE_AA
    )

    # Add count with outline
    count_text = f"Candidates: {len(contours)}"
    cv2.putText(
        overlay, count_text, (20, DEBUG_SUBTITLE_Y),
        DEBUG_FONT_FACE, DEBUG_SUBTITLE_FONT_SCALE, DEBUG_COLOR_WHITE,
        DEBUG_SUBTITLE_OUTLINE_THICKNESS, cv2.LINE_AA
    )
    cv2.putText(
        overlay, count_text, (20, DEBUG_SUBTITLE_Y),
        DEBUG_FONT_FACE, DEBUG_SUBTITLE_FONT_SCALE, color,
        DEBUG_SUBTITLE_THICKNESS, cv2.LINE_AA
    )

    return overlay


def draw_candidates_with_scores(
    image: np.ndarray,
    candidates: List[Tuple[np.ndarray, float, Dict[str, Any]]],
    title: str,
) -> np.ndarray:
    """
    Draw candidate contours with scores and details.

    Args:
        image: Original image
        candidates: List of (corners, score, details) tuples
        title: Title for the visualization

    Returns:
        Annotated image
    """
    overlay = image.copy()

    # Color palette for candidates (different colors for ranking)
    colors = [
        DEBUG_COLOR_GREEN,    # Green - best
        DEBUG_COLOR_YELLOW,   # Yellow
        DEBUG_COLOR_ORANGE,   # Orange
        DEBUG_COLOR_MAGENTA,  # Magenta
        DEBUG_COLOR_PINK      # Pink
    ]

    for idx, (corners, score, details) in enumerate(candidates):
        color = colors[idx % len(colors)]

        # Draw quadrilateral
        pts = corners.reshape(4, 2).astype(np.int32)
        cv2.polylines(overlay, [pts], True, color, DEBUG_CONTOUR_THICKNESS)

        # Draw corner circles
        for pt in pts:
            cv2.circle(overlay, tuple(pt), DEBUG_CORNER_RADIUS, color, -1)

        # Prepare annotation text
        if score > 0:
            aspect_ratio = details.get("aspect_ratio", 0)
            area_ratio = details.get("area", 0) / (image.shape[0] * image.shape[1])
            text = f"#{idx+1} Score:{score:.2f} AR:{aspect_ratio:.2f} Area:{area_ratio:.2%}"
        else:
            reject_reason = details.get("reject_reason", "unknown")
            text = f"#{idx+1} REJECT: {reject_reason}"

        # Position text near first corner
        text_pos = (int(pts[0][0]) + 10, int(pts[0][1]) - 10)

        # Draw text with outline for visibility
        cv2.putText(
            overlay, text, text_pos,
            DEBUG_FONT_FACE, DEBUG_LABEL_FONT_SCALE, DEBUG_COLOR_BLACK,
            DEBUG_LABEL_OUTLINE_THICKNESS, cv2.LINE_AA
        )
        cv2.putText(
            overlay, text, text_pos,
            DEBUG_FONT_FACE, DEBUG_LABEL_FONT_SCALE, color,
            DEBUG_LABEL_THICKNESS, cv2.LINE_AA
        )

    # Add title with outline
    cv2.putText(
        overlay, title, (20, DEBUG_TITLE_Y),
        DEBUG_FONT_FACE, DEBUG_TITLE_FONT_SCALE, DEBUG_COLOR_WHITE,
        DEBUG_TITLE_OUTLINE_THICKNESS, cv2.LINE_AA
    )
    cv2.putText(
        overlay, title, (20, DEBUG_TITLE_Y),
        DEBUG_FONT_FACE, DEBUG_TITLE_FONT_SCALE, DEBUG_COLOR_CYAN,
        DEBUG_TITLE_THICKNESS, cv2.LINE_AA
    )

    return overlay


def refine_corners(gray: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]:
    """
    Refine corner positions to sub-pixel accuracy using cornerSubPix.

    Note: Credit cards have rounded corners (~3mm radius), so perfect corner
    detection is inherently limited. This provides best-effort refinement.

    Args:
        gray: Grayscale image
        corners: Initial corner positions (4x1x2 or 4x2 array)

    Returns:
        Refined corners or None if refinement fails
    """
    try:
        corners_float = corners.reshape(-1, 1, 2).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Use larger search window (11x11) to handle rounded corners better
        refined = cv2.cornerSubPix(gray, corners_float, (11, 11), (-1, -1), criteria)
        return refined
    except:
        return None


def find_card_contours(image: np.ndarray, debug_dir: Optional[str] = None) -> List[np.ndarray]:
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

    # Save original image
    save_debug_image(image, "01_original.png", debug_dir)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_debug_image(gray, "02_grayscale.png", debug_dir)

    # Apply bilateral filter to reduce noise while keeping edges
    filtered = cv2.bilateralFilter(gray, 11, 75, 75)
    save_debug_image(filtered, "03_bilateral_filtered.png", debug_dir)

    candidates = []
    canny_candidates = []
    adaptive_candidates = []
    otsu_candidates = []
    color_candidates = []

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
                    # Refine corners to sub-pixel accuracy
                    refined = refine_corners(gray, approx)
                    quads.append(refined if refined is not None else approx)
                    break
                elif len(approx) > 4 and len(approx) <= 8:
                    # Try to find 4 dominant corners using convex hull
                    hull = cv2.convexHull(approx)
                    if len(hull) >= 4:
                        # Get the 4 corners with maximum area
                        hull_approx = cv2.approxPolyDP(hull, 0.05 * cv2.arcLength(hull, True), True)
                        if len(hull_approx) == 4:
                            # Refine corners to sub-pixel accuracy
                            refined = refine_corners(gray, hull_approx)
                            quads.append(refined if refined is not None else hull_approx)
                            break

        return quads

    # Strategy 1: Canny edge detection with various thresholds
    canny_configs = [(20, 60), (30, 100), (50, 150), (75, 200), (100, 250)]
    saved_canny_indices = [0, 2, 4]  # Save representative samples

    for idx, (canny_low, canny_high) in enumerate(canny_configs):
        edges = cv2.Canny(filtered, canny_low, canny_high)

        # Save representative edge images
        if idx in saved_canny_indices and debug_dir:
            save_debug_image(edges, f"04_canny_{canny_low}_{canny_high}.png", debug_dir)

        # Morphological closing to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges_morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Save morphology result for best threshold
        if idx == 2 and debug_dir:
            save_debug_image(edges_morphed, "07_canny_morphology.png", debug_dir)

        # Find all contours (not just external)
        contours, _ = cv2.findContours(edges_morphed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        quads = extract_quads(contours)
        candidates.extend(quads)
        canny_candidates.extend(quads)

    # Save Canny contours overlay
    if debug_dir and canny_candidates:
        canny_overlay = draw_contours_overlay(image, canny_candidates, "Canny Edge Detection", (0, 255, 255))
        save_debug_image(canny_overlay, "08_canny_contours.png", debug_dir)

    # Strategy 2: Adaptive thresholding (for varying lighting)
    adaptive_configs = [(11, 2), (21, 5), (31, 10), (51, 10)]
    saved_adaptive = [0, 2]  # Save representative samples

    for idx, (block_size, C) in enumerate(adaptive_configs):
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C
        )

        # Save representative thresholded images
        if idx in saved_adaptive and debug_dir:
            if idx == 0:
                save_debug_image(thresh, "09_adaptive_11_2.png", debug_dir)
            elif idx == 2:
                save_debug_image(thresh, "10_adaptive_31_10.png", debug_dir)

        # Invert if needed
        for img in [thresh, 255 - thresh]:
            contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            quads = extract_quads(contours)
            candidates.extend(quads)
            adaptive_candidates.extend(quads)

    # Save adaptive contours overlay
    if debug_dir and adaptive_candidates:
        adaptive_overlay = draw_contours_overlay(image, adaptive_candidates, "Adaptive Thresholding", (255, 128, 0))
        save_debug_image(adaptive_overlay, "11_adaptive_contours.png", debug_dir)

    # Strategy 3: Otsu's thresholding
    _, otsu = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_debug_image(otsu, "12_otsu_binary.png", debug_dir)

    otsu_inverted = 255 - otsu
    save_debug_image(otsu_inverted, "13_otsu_inverted.png", debug_dir)

    for img in [otsu, otsu_inverted]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_morphed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(img_morphed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        quads = extract_quads(contours)
        candidates.extend(quads)
        otsu_candidates.extend(quads)

    # Save Otsu contours overlay
    if debug_dir and otsu_candidates:
        otsu_overlay = draw_contours_overlay(image, otsu_candidates, "Otsu Thresholding", (255, 0, 255))
        save_debug_image(otsu_overlay, "14_otsu_contours.png", debug_dir)

    # Strategy 4: Color-based segmentation (gray card on light background)
    # Look for regions that are darker than the background
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]  # Saturation channel
    save_debug_image(sat, "15_hsv_saturation.png", debug_dir)

    # Low saturation = gray colors (like metallic card)
    _, low_sat_mask = cv2.threshold(sat, 30, 255, cv2.THRESH_BINARY_INV)
    save_debug_image(low_sat_mask, "16_low_sat_mask.png", debug_dir)

    # Combine with value channel to find gray regions
    val = hsv[:, :, 2]
    gray_mask = cv2.bitwise_and(low_sat_mask, cv2.inRange(val, 80, 200))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)
    save_debug_image(gray_mask, "17_gray_mask.png", debug_dir)

    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads = extract_quads(contours, epsilon_factor=0.03)
    candidates.extend(quads)
    color_candidates.extend(quads)

    # Save color-based contours overlay
    if debug_dir and color_candidates:
        color_overlay = draw_contours_overlay(image, color_candidates, "Color-Based Detection", (0, 255, 0))
        save_debug_image(color_overlay, "18_color_contours.png", debug_dir)

    # Save all candidates overlay
    if debug_dir and candidates:
        all_overlay = draw_contours_overlay(image, candidates, "All Candidates", (128, 128, 255))
        save_debug_image(all_overlay, "19_all_candidates.png", debug_dir)

    return candidates


def detect_credit_card(
    image: np.ndarray,
    aspect_ratio_tolerance: float = 0.15,
    debug_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Detect a credit card in the image.

    Args:
        image: Input BGR image
        aspect_ratio_tolerance: Allowed deviation from standard aspect ratio
        debug_dir: Optional directory to save debug images

    Returns:
        Dictionary containing:
        - corners: 4x2 array of corner points (ordered)
        - contour: Full contour points
        - confidence: Detection confidence score
        - width_px, height_px: Detected dimensions
        - aspect_ratio: Detected aspect ratio
        Or None if no card detected
    """
    if debug_dir:
        print(f"Saving card detection debug images to: {debug_dir}")

    h, w = image.shape[:2]
    image_area = h * w

    # Find candidate contours
    candidates = find_card_contours(image, debug_dir=debug_dir)

    if not candidates:
        if debug_dir:
            print("  No candidates found")
        return None

    # Score each candidate
    best_score = 0.0
    best_result = None
    all_scored = []

    for contour in candidates:
        corners = contour.reshape(4, 2)
        score, details = score_card_candidate(
            contour, corners, image_area, aspect_ratio_tolerance
        )

        all_scored.append((corners, score, details))

        if score > best_score:
            best_score = score
            best_result = details

    # Sort by score (descending) and take top 5
    all_scored.sort(key=lambda x: x[1], reverse=True)
    top_candidates = all_scored[:5]

    # Save scored candidates visualization
    if debug_dir and top_candidates:
        scored_overlay = draw_candidates_with_scores(image, top_candidates, "Top 5 Candidates")
        save_debug_image(scored_overlay, "20_scored_candidates.png", debug_dir)

    if best_result is None or best_score < 0.3:
        if debug_dir:
            print(f"  Best score {best_score:.2f} below threshold 0.3")
        return None

    # Save final detection
    if debug_dir:
        final_overlay = image.copy()
        corners = best_result["corners"].astype(np.int32)
        cv2.polylines(final_overlay, [corners], True, DEBUG_COLOR_GREEN, DEBUG_CONTOUR_THICKNESS + 1)

        # Draw corners
        for pt in corners:
            cv2.circle(final_overlay, tuple(pt), DEBUG_CORNER_RADIUS + 2, (0, 0, 255), -1)

        # Add details text
        text_y = DEBUG_TITLE_Y
        details_text = [
            "Final Detection",
            f"Score: {best_score:.3f}",
            f"Aspect Ratio: {best_result['aspect_ratio']:.3f}",
            f"Dimensions: {best_result['width']:.0f}x{best_result['height']:.0f}px",
        ]

        for text in details_text:
            cv2.putText(
                final_overlay, text, (20, text_y),
                DEBUG_FONT_FACE, DEBUG_SUBTITLE_FONT_SCALE, DEBUG_COLOR_WHITE,
                DEBUG_SUBTITLE_OUTLINE_THICKNESS, cv2.LINE_AA
            )
            cv2.putText(
                final_overlay, text, (20, text_y),
                DEBUG_FONT_FACE, DEBUG_SUBTITLE_FONT_SCALE, DEBUG_COLOR_GREEN,
                DEBUG_SUBTITLE_THICKNESS, cv2.LINE_AA
            )
            text_y += DEBUG_LINE_SPACING

        save_debug_image(final_overlay, "21_final_detection.png", debug_dir)
        print(f"  Saved 21 debug images")

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
