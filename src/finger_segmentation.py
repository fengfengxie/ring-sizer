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
from typing import Optional, Dict, Any, Literal, List, Tuple
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
from pathlib import Path

# Import visualization constants
from src.viz_constants import (
    FONT_FACE, FontScale, FontThickness, Color, Size, Layout
)

FingerIndex = Literal["auto", "index", "middle", "ring", "pinky"]

# MediaPipe hand landmark indices for each finger
# Each finger has 4 landmarks: MCP (knuckle), PIP, DIP, TIP
FINGER_LANDMARKS = {
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

# Thumb landmarks (special case - not typically used for ring measurement)
THUMB_LANDMARKS = [1, 2, 3, 4]

# Wrist landmark
WRIST_LANDMARK = 0

# Palm landmarks (for creating hand mask)
PALM_LANDMARKS = [0, 1, 5, 9, 13, 17]

# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Initialize MediaPipe Hands (lazy loading)
_hands_detector = None

# Hand skeleton connections (MediaPipe hand model)
HAND_CONNECTIONS = [
    # Palm
    (0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index
    (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
]

# Finger colors for visualization
FINGER_COLORS = {
    "thumb": Color.RED,
    "index": Color.CYAN,
    "middle": Color.YELLOW,
    "ring": Color.MAGENTA,
    "pinky": Color.ORANGE,
}


def save_debug_image(image: np.ndarray, filename: str, debug_dir: Optional[str]) -> None:
    """
    Save debug image with compression and downsampling.

    Args:
        image: Image to save
        filename: Output filename
        debug_dir: Directory to save to (if None, skip saving)
    """
    if debug_dir is None:
        return

    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(debug_dir) / filename

    # Downsample if too large (max 1920px dimension)
    h, w = image.shape[:2]
    max_dim = 1920
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # PNG compression
    cv2.imwrite(str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def draw_landmarks_overlay(image: np.ndarray, landmarks: np.ndarray, label: bool = True) -> np.ndarray:
    """
    Draw hand landmarks as numbered circles.

    Args:
        image: Input image
        landmarks: 21x2 array of landmark positions
        label: Whether to draw landmark numbers

    Returns:
        Image with landmarks drawn
    """
    overlay = image.copy()

    for i, (x, y) in enumerate(landmarks):
        # Draw circle
        cv2.circle(overlay, (int(x), int(y)), Size.ENDPOINT_RADIUS, Color.GREEN, -1)
        cv2.circle(overlay, (int(x), int(y)), Size.ENDPOINT_RADIUS, Color.BLACK, 2)

        # Draw number
        if label:
            text = str(i)
            text_size = cv2.getTextSize(text, FONT_FACE, FontScale.SMALL, FontThickness.BODY)[0]
            text_x = int(x - text_size[0] / 2)
            text_y = int(y + text_size[1] / 2)

            # Black outline
            cv2.putText(overlay, text, (text_x, text_y), FONT_FACE, FontScale.SMALL,
                       Color.BLACK, FontThickness.BODY + 2, cv2.LINE_AA)
            # White text
            cv2.putText(overlay, text, (text_x, text_y), FONT_FACE, FontScale.SMALL,
                       Color.WHITE, FontThickness.BODY, cv2.LINE_AA)

    return overlay


def draw_hand_skeleton(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Draw hand skeleton with connections between landmarks.

    Args:
        image: Input image
        landmarks: 21x2 array of landmark positions

    Returns:
        Image with skeleton drawn
    """
    overlay = image.copy()

    # Draw connections
    for idx1, idx2 in HAND_CONNECTIONS:
        pt1 = (int(landmarks[idx1, 0]), int(landmarks[idx1, 1]))
        pt2 = (int(landmarks[idx2, 0]), int(landmarks[idx2, 1]))
        cv2.line(overlay, pt1, pt2, Color.CYAN, Size.LINE_THICK, cv2.LINE_AA)

    # Draw landmarks on top
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(overlay, (int(x), int(y)), Size.CORNER_RADIUS, Color.GREEN, -1)
        cv2.circle(overlay, (int(x), int(y)), Size.CORNER_RADIUS, Color.BLACK, 2)

    return overlay


def draw_detection_info(image: np.ndarray, confidence: float, handedness: str, rotation: int) -> np.ndarray:
    """
    Draw detection metadata on image.

    Args:
        image: Input image
        confidence: Detection confidence (0-1)
        handedness: "Left" or "Right"
        rotation: Rotation code (0, 1, 2, 3)

    Returns:
        Image with text overlay
    """
    overlay = image.copy()

    rotation_names = {0: "None", 1: "90° CW", 2: "180°", 3: "90° CCW"}
    rotation_name = rotation_names.get(rotation, "Unknown")

    lines = [
        f"Confidence: {confidence:.3f}",
        f"Hand: {handedness}",
        f"Rotation: {rotation_name}",
    ]

    y = Layout.TITLE_Y
    for line in lines:
        # Black outline
        cv2.putText(overlay, line, (Layout.TEXT_OFFSET_X, y), FONT_FACE, FontScale.BODY,
                   Color.BLACK, FontThickness.LABEL_OUTLINE, cv2.LINE_AA)
        # White text
        cv2.putText(overlay, line, (Layout.TEXT_OFFSET_X, y), FONT_FACE, FontScale.BODY,
                   Color.WHITE, FontThickness.LABEL, cv2.LINE_AA)
        y += Layout.LINE_SPACING

    return overlay


def draw_finger_regions(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Draw individual finger regions in different colors.

    Args:
        image: Input image
        landmarks: 21x2 array of landmark positions

    Returns:
        Image with colored finger regions
    """
    h, w = image.shape[:2]
    overlay = image.copy()
    mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw thumb
    thumb_pts = landmarks[THUMB_LANDMARKS].astype(np.int32)
    cv2.fillConvexPoly(mask_overlay, thumb_pts, FINGER_COLORS["thumb"])

    # Draw each finger
    for finger_name, indices in FINGER_LANDMARKS.items():
        finger_pts = landmarks[indices].astype(np.int32)
        cv2.fillConvexPoly(mask_overlay, finger_pts, FINGER_COLORS[finger_name])

    # Blend with original
    overlay = cv2.addWeighted(overlay, 0.6, mask_overlay, 0.4, 0)

    return overlay


def draw_extension_scores(image: np.ndarray, scores: Dict[str, float], selected: str) -> np.ndarray:
    """
    Draw finger extension scores.

    Args:
        image: Input image
        scores: Dict mapping finger name to extension score
        selected: Name of selected finger

    Returns:
        Image with scores drawn
    """
    overlay = image.copy()

    # Sort by score
    sorted_fingers = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    y = Layout.TITLE_Y
    for finger_name, score in sorted_fingers:
        is_selected = (finger_name == selected)
        color = Color.GREEN if is_selected else Color.WHITE
        text = f"{finger_name.capitalize()}: {score:.1f}" + (" ✓" if is_selected else "")

        # Black outline
        cv2.putText(overlay, text, (Layout.TEXT_OFFSET_X, y), FONT_FACE, FontScale.BODY,
                   Color.BLACK, FontThickness.LABEL_OUTLINE, cv2.LINE_AA)
        # Colored text
        cv2.putText(overlay, text, (Layout.TEXT_OFFSET_X, y), FONT_FACE, FontScale.BODY,
                   color, FontThickness.LABEL, cv2.LINE_AA)
        y += Layout.LINE_SPACING

    return overlay


def draw_component_stats(image: np.ndarray, labels: np.ndarray, stats: np.ndarray,
                         selected_idx: int) -> np.ndarray:
    """
    Draw connected component statistics.

    Args:
        image: Input image
        labels: Connected component labels
        stats: Component statistics from cv2.connectedComponentsWithStats
        selected_idx: Index of selected component

    Returns:
        Image with colored components and stats
    """
    overlay = image.copy()

    # Create colored component visualization
    num_labels = stats.shape[0]
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black
    colors[selected_idx] = Color.GREEN  # Selected is green

    colored = colors[labels]
    overlay = cv2.addWeighted(overlay, 0.5, colored, 0.5, 0)

    # Draw text stats
    y = Layout.TITLE_Y
    lines = [
        f"Components: {num_labels - 1}",  # Exclude background
        f"Selected area: {stats[selected_idx, cv2.CC_STAT_AREA]} px",
    ]

    for line in lines:
        cv2.putText(overlay, line, (Layout.TEXT_OFFSET_X, y), FONT_FACE, FontScale.BODY,
                   Color.BLACK, FontThickness.LABEL_OUTLINE, cv2.LINE_AA)
        cv2.putText(overlay, line, (Layout.TEXT_OFFSET_X, y), FONT_FACE, FontScale.BODY,
                   Color.WHITE, FontThickness.LABEL, cv2.LINE_AA)
        y += Layout.LINE_SPACING

    return overlay


def _download_model():
    """Download the hand landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print(f"Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model downloaded to {MODEL_PATH}")


def _get_hands_detector(force_new: bool = False):
    """Get or initialize the MediaPipe Hands detector."""
    global _hands_detector
    if _hands_detector is None or force_new:
        _download_model()
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.3,  # Lower threshold for better detection
            min_tracking_confidence=0.3,
        )
        _hands_detector = vision.HandLandmarker.create_from_options(options)
    return _hands_detector


def _try_detect_hand(detector, image: np.ndarray) -> Optional[Tuple[Any, int]]:
    """
    Try to detect hand in image, returns (results, rotation_code) or None.
    rotation_code: 0=none, 1=90cw, 2=180, 3=90ccw
    """
    # Try different rotations to handle various image orientations
    rotations = [
        (image, 0),
        (cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 1),
        (cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 3),
        (cv2.rotate(image, cv2.ROTATE_180), 2),
    ]

    best_result = None
    best_confidence = 0
    best_rotation = 0

    for rotated, rot_code in rotations:
        # Convert to RGB and ensure contiguous memory layout
        rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)

        if results.hand_landmarks:
            # Get best confidence among detected hands
            for i, handedness in enumerate(results.handedness):
                conf = handedness[0].score
                if conf > best_confidence:
                    best_confidence = conf
                    best_result = results
                    best_rotation = rot_code

    if best_result is None:
        return None

    return best_result, best_rotation


def _transform_landmarks_for_rotation(
    landmarks: np.ndarray,
    rotation_code: int,
    original_h: int,
    original_w: int,
) -> np.ndarray:
    """
    Transform landmarks from rotated coordinates back to original image coordinates.
    """
    if rotation_code == 0:
        # No rotation
        return landmarks
    elif rotation_code == 1:
        # Was rotated 90 CW, so transform back (90 CCW)
        # In rotated: (x, y) with size (h, w) -> original: (y, w-1-x) with size (w, h)
        new_landmarks = np.zeros_like(landmarks)
        new_landmarks[:, 0] = landmarks[:, 1] * original_w  # y -> x
        new_landmarks[:, 1] = (1 - landmarks[:, 0]) * original_h  # (1-x) -> y
        return new_landmarks
    elif rotation_code == 2:
        # Was rotated 180
        new_landmarks = np.zeros_like(landmarks)
        new_landmarks[:, 0] = (1 - landmarks[:, 0]) * original_w
        new_landmarks[:, 1] = (1 - landmarks[:, 1]) * original_h
        return new_landmarks
    elif rotation_code == 3:
        # Was rotated 90 CCW, so transform back (90 CW)
        new_landmarks = np.zeros_like(landmarks)
        new_landmarks[:, 0] = (1 - landmarks[:, 1]) * original_w
        new_landmarks[:, 1] = landmarks[:, 0] * original_h
        return new_landmarks

    return landmarks


def segment_hand(
    image: np.ndarray,
    max_dimension: int = 1280,
    debug_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Detect and segment hand from image using MediaPipe.

    Args:
        image: Input BGR image
        max_dimension: Maximum dimension for processing (large images are resized)
        debug_dir: Optional directory to save debug images

    Returns:
        Dictionary containing:
        - landmarks: 21x2 array of landmark positions (pixel coordinates)
        - landmarks_normalized: 21x2 array of normalized coordinates [0-1]
        - mask: Binary hand mask
        - confidence: Detection confidence
        - handedness: "Left" or "Right"
        Or None if no hand detected
    """
    h, w = image.shape[:2]

    # Debug: Save original image
    save_debug_image(image, "01_original.png", debug_dir)

    # Resize if image is too large (MediaPipe works better with smaller images)
    scale = 1.0
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = image
        new_h, new_w = h, w

    # Debug: Save resized image (if resized)
    if scale != 1.0:
        save_debug_image(resized, "02_resized_for_detection.png", debug_dir)

    # Process with MediaPipe (try multiple rotations)
    detector = _get_hands_detector()
    detection_result = _try_detect_hand(detector, resized)

    if detection_result is None:
        return None

    results, rotation_code = detection_result

    # Select the best hand (highest confidence)
    best_hand_idx = 0
    best_conf = 0
    for i, handedness in enumerate(results.handedness):
        if handedness[0].score > best_conf:
            best_conf = handedness[0].score
            best_hand_idx = i

    hand_landmarks = results.hand_landmarks[best_hand_idx]
    handedness = results.handedness[best_hand_idx]

    # Extract landmark coordinates (normalized 0-1 in rotated image)
    landmarks_normalized_rotated = np.array([
        [lm.x, lm.y] for lm in hand_landmarks
    ])

    # Transform landmarks back to original orientation
    landmarks = _transform_landmarks_for_rotation(
        landmarks_normalized_rotated, rotation_code, h, w
    )

    # Compute normalized landmarks for original orientation
    landmarks_normalized = landmarks.copy()
    landmarks_normalized[:, 0] /= w
    landmarks_normalized[:, 1] /= h

    # Debug: Draw landmarks overlay
    if debug_dir:
        landmarks_img = draw_landmarks_overlay(image, landmarks, label=True)
        save_debug_image(landmarks_img, "03_landmarks_overlay.png", debug_dir)

        # Draw hand skeleton
        skeleton_img = draw_hand_skeleton(image, landmarks)
        save_debug_image(skeleton_img, "04_hand_skeleton.png", debug_dir)

        # Draw detection info
        info_img = draw_detection_info(image, handedness[0].score,
                                       handedness[0].category_name, rotation_code)
        save_debug_image(info_img, "05_detection_info.png", debug_dir)

    # Generate hand mask at original resolution
    mask = _create_hand_mask(landmarks, (h, w), debug_dir=debug_dir)

    return {
        "landmarks": landmarks,
        "landmarks_normalized": landmarks_normalized,
        "mask": mask,
        "confidence": handedness[0].score,
        "handedness": handedness[0].category_name,
        "rotation_applied": rotation_code,
    }


def _create_hand_mask(landmarks: np.ndarray, shape: Tuple[int, int],
                      debug_dir: Optional[str] = None) -> np.ndarray:
    """
    Create a binary mask of the hand region from landmarks.

    Args:
        landmarks: 21x2 array of landmark pixel coordinates
        shape: (height, width) of output mask
        debug_dir: Optional directory to save debug images

    Returns:
        Binary mask (uint8, 0 or 255)
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Debug: Create reference image for overlays
    if debug_dir:
        # Create a color image from mask for visualization
        ref_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Create convex hull of all landmarks
    hull_points = cv2.convexHull(landmarks.astype(np.int32))
    cv2.fillConvexPoly(mask, hull_points, 255)

    # Debug: Save convex hull mask
    if debug_dir:
        hull_img = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
        # Draw hull outline
        cv2.polylines(hull_img, [hull_points], True, Color.GREEN, Size.CONTOUR_THICK)
        save_debug_image(hull_img, "07_convex_hull.png", debug_dir)

    # Also fill individual finger regions for better coverage
    finger_mask = mask.copy()
    for finger_name, indices in FINGER_LANDMARKS.items():
        finger_pts = landmarks[indices].astype(np.int32)
        # Create a polygon along the finger
        cv2.fillConvexPoly(finger_mask, finger_pts, 255)

    # Fill thumb
    thumb_pts = landmarks[THUMB_LANDMARKS].astype(np.int32)
    cv2.fillConvexPoly(finger_mask, thumb_pts, 255)

    # Debug: Save finger regions visualization
    if debug_dir:
        # Create colored finger regions using existing helper
        finger_color_img = np.zeros((h, w, 3), dtype=np.uint8)
        for finger_name, indices in FINGER_LANDMARKS.items():
            finger_pts = landmarks[indices].astype(np.int32)
            cv2.fillConvexPoly(finger_color_img, finger_pts, FINGER_COLORS[finger_name])
        cv2.fillConvexPoly(finger_color_img, thumb_pts, FINGER_COLORS["thumb"])
        save_debug_image(finger_color_img, "08_finger_regions.png", debug_dir)

        # Save raw combined mask
        save_debug_image(finger_mask, "09_raw_hand_mask.png", debug_dir)

    mask = finger_mask

    # Apply morphological operations to smooth the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # Morphological closing (fill gaps)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if debug_dir:
        save_debug_image(mask, "10_morph_close.png", debug_dir)

    # Morphological opening (remove noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if debug_dir:
        save_debug_image(mask, "11_morph_open.png", debug_dir)

        # Save final mask with semi-transparent overlay
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # Create green tint for final mask
        mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        mask_overlay[mask > 0] = Color.GREEN
        save_debug_image(mask_overlay, "12_final_hand_mask.png", debug_dir)

    return mask


def _calculate_finger_extension(landmarks: np.ndarray, finger_indices: List[int]) -> float:
    """
    Calculate how extended a finger is based on landmark positions.

    Returns a score where higher = more extended.
    """
    if len(finger_indices) < 4:
        return 0.0

    # Get finger landmarks
    mcp = landmarks[finger_indices[0]]  # Knuckle
    pip = landmarks[finger_indices[1]]  # First joint
    dip = landmarks[finger_indices[2]]  # Second joint
    tip = landmarks[finger_indices[3]]  # Fingertip

    # Calculate vectors
    mcp_to_tip = tip - mcp
    mcp_to_pip = pip - mcp

    # Extension score based on:
    # 1. Distance from knuckle to tip (longer = more extended)
    finger_length = np.linalg.norm(mcp_to_tip)

    # 2. Straightness (how aligned are the joints)
    pip_to_dip = dip - pip
    dip_to_tip = tip - dip

    # Dot products to check alignment (1 = straight, -1 = bent back)
    if np.linalg.norm(mcp_to_pip) > 0 and np.linalg.norm(pip_to_dip) > 0:
        align1 = np.dot(mcp_to_pip, pip_to_dip) / (np.linalg.norm(mcp_to_pip) * np.linalg.norm(pip_to_dip))
    else:
        align1 = 0

    if np.linalg.norm(pip_to_dip) > 0 and np.linalg.norm(dip_to_tip) > 0:
        align2 = np.dot(pip_to_dip, dip_to_tip) / (np.linalg.norm(pip_to_dip) * np.linalg.norm(dip_to_tip))
    else:
        align2 = 0

    straightness = (align1 + align2) / 2

    # Combined score
    return finger_length * (0.5 + 0.5 * max(0, straightness))


def _create_finger_roi_mask(
    finger_landmarks: np.ndarray,
    all_landmarks: np.ndarray,
    shape: Tuple[int, int],
    expansion_factor: float = 1.8,
) -> np.ndarray:
    """
    Create a Region of Interest (ROI) mask around finger landmarks.

    This creates a generous bounding region that should contain the entire finger
    without cutting off edges, but excludes other fingers.

    Args:
        finger_landmarks: 4x2 array of finger landmark positions (MCP, PIP, DIP, TIP)
        all_landmarks: 21x2 array of all hand landmarks
        shape: (height, width) of output mask
        expansion_factor: How much to expand perpendicular to finger axis

    Returns:
        Binary ROI mask
    """
    h, w = shape
    roi_mask = np.zeros((h, w), dtype=np.uint8)

    # Calculate finger axis direction
    mcp = finger_landmarks[0]
    tip = finger_landmarks[3]
    finger_axis = tip - mcp
    finger_length = np.linalg.norm(finger_axis)

    if finger_length < 1:
        return roi_mask

    finger_direction = finger_axis / finger_length

    # Perpendicular direction
    perp = np.array([-finger_direction[1], finger_direction[0]])

    # Estimate finger width from landmark spacing
    # Use median distance between consecutive landmarks as width proxy
    segment_lengths = []
    for i in range(len(finger_landmarks) - 1):
        seg_len = np.linalg.norm(finger_landmarks[i + 1] - finger_landmarks[i])
        segment_lengths.append(seg_len)
    avg_segment = np.median(segment_lengths) if segment_lengths else finger_length / 3

    # Finger width is roughly 1/3 to 1/2 of segment length
    base_width = avg_segment * 0.6 * expansion_factor

    # Extend ROI slightly beyond landmarks (towards palm and beyond tip)
    wrist = all_landmarks[WRIST_LANDMARK]
    palm_direction = mcp - wrist
    palm_direction = palm_direction / (np.linalg.norm(palm_direction) + 1e-8)

    # Extend 20% beyond MCP toward palm
    extended_base = mcp - palm_direction * finger_length * 0.2
    # Extend 10% beyond tip
    extended_tip = tip + finger_direction * finger_length * 0.1

    # Create polygon along finger with wider margins
    polygon_points = []
    num_samples = 8  # More points for smoother ROI

    for i in range(num_samples):
        t = i / (num_samples - 1)
        # Interpolate from extended base to extended tip
        pt = extended_base + (extended_tip - extended_base) * t

        # Width varies: wider at base, narrower at tip
        width_scale = 1.0 - 0.2 * t
        half_width = base_width * width_scale / 2

        # Add left and right points
        left = pt + perp * half_width
        right = pt - perp * half_width
        polygon_points.append((left, right))

    # Build polygon
    polygon = []
    for left, right in polygon_points:
        polygon.append(left)
    for left, right in reversed(polygon_points):
        polygon.append(right)

    polygon = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(roi_mask, [polygon], 255)

    return roi_mask


def _isolate_finger_from_hand_mask(
    hand_mask: np.ndarray,
    finger_landmarks: np.ndarray,
    all_landmarks: np.ndarray,
    min_area: int = 500,
) -> Optional[np.ndarray]:
    """
    Isolate finger using pixel-level intersection of hand mask with finger ROI.

    This is the preferred method as it preserves actual finger edges from MediaPipe
    rather than creating a synthetic polygon.

    Args:
        hand_mask: Full hand mask from MediaPipe (pixel-accurate)
        finger_landmarks: 4x2 array of finger landmarks
        all_landmarks: 21x2 array of all hand landmarks
        min_area: Minimum valid finger area

    Returns:
        Binary finger mask, or None if isolation fails
    """
    h, w = hand_mask.shape

    # Create ROI mask around finger
    roi_mask = _create_finger_roi_mask(finger_landmarks, all_landmarks, (h, w))

    # Intersect hand mask with finger ROI
    # This preserves real pixel-level edges from MediaPipe
    finger_mask = cv2.bitwise_and(hand_mask, roi_mask)

    # Find connected components to remove fragments from other fingers
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        finger_mask, connectivity=8
    )

    if num_labels <= 1:
        return None

    # Select component closest to finger landmarks centroid
    landmarks_centroid = np.mean(finger_landmarks, axis=0)

    best_component = None
    best_distance = float('inf')

    for i in range(1, num_labels):  # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        component_centroid = centroids[i]
        dist = np.linalg.norm(component_centroid - landmarks_centroid)

        if dist < best_distance:
            best_distance = dist
            best_component = i

    if best_component is None:
        return None

    # Create final mask with only the selected component
    final_mask = np.zeros_like(finger_mask)
    final_mask[labels == best_component] = 255

    return final_mask


def isolate_finger(
    hand_data: Dict[str, Any],
    finger: FingerIndex = "auto",
    image_shape: Optional[Tuple[int, int]] = None,
    image: Optional[np.ndarray] = None,
    debug_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Isolate a specific finger from hand segmentation data.

    Args:
        hand_data: Output from segment_hand()
        finger: Which finger to isolate, or "auto" to select most extended
        image_shape: (height, width) for mask generation
        image: Optional original image for debug visualization
        debug_dir: Optional directory to save debug images

    Returns:
        Dictionary containing:
        - mask: Binary finger mask
        - landmarks: Finger landmark positions (4x2 array)
        - base_point: Palm-side base of finger (MCP joint)
        - tip_point: Fingertip position
        - finger_name: Name of the isolated finger
        Or None if finger cannot be isolated
    """
    landmarks = hand_data["landmarks"]

    if image_shape is None:
        # Estimate from hand mask
        if "mask" in hand_data:
            image_shape = hand_data["mask"].shape[:2]
        else:
            return None

    # Determine which finger to use
    extension_scores = {}
    if finger == "auto":
        # Find the most extended finger
        best_finger = None
        best_score = -1

        for finger_name, indices in FINGER_LANDMARKS.items():
            score = _calculate_finger_extension(landmarks, indices)
            extension_scores[finger_name] = score
            if score > best_score:
                best_score = score
                best_finger = finger_name

        if best_finger is None:
            return None
        finger = best_finger

        # Debug: Draw extension scores
        if debug_dir and image is not None:
            scores_img = draw_extension_scores(image, extension_scores, finger)
            save_debug_image(scores_img, "13_finger_extension_scores.png", debug_dir)

    if finger not in FINGER_LANDMARKS:
        return None

    indices = FINGER_LANDMARKS[finger]
    finger_landmarks = landmarks[indices]

    # Debug: Highlight selected finger landmarks
    if debug_dir and image is not None:
        selected_img = image.copy()
        for i, (x, y) in enumerate(finger_landmarks):
            color = Color.GREEN if i == 0 else (Color.YELLOW if i == 3 else Color.CYAN)
            cv2.circle(selected_img, (int(x), int(y)), Size.ENDPOINT_RADIUS, color, -1)
            cv2.circle(selected_img, (int(x), int(y)), Size.ENDPOINT_RADIUS, Color.BLACK, 2)

            # Label landmarks
            labels = ["MCP", "PIP", "DIP", "TIP"]
            cv2.putText(selected_img, labels[i], (int(x) + 20, int(y)), FONT_FACE,
                       FontScale.SMALL, Color.BLACK, FontThickness.BODY + 2, cv2.LINE_AA)
            cv2.putText(selected_img, labels[i], (int(x) + 20, int(y)), FONT_FACE,
                       FontScale.SMALL, Color.WHITE, FontThickness.BODY, cv2.LINE_AA)
        save_debug_image(selected_img, "14_selected_finger_landmarks.png", debug_dir)

    # Create finger mask using pixel-level approach (preferred)
    # This preserves actual finger edges from MediaPipe hand segmentation
    mask_pixel = None
    mask_polygon = None
    method_used = "unknown"

    if "mask" in hand_data and hand_data["mask"] is not None:
        # Try pixel-level approach first (more accurate)
        mask_pixel = _isolate_finger_from_hand_mask(
            hand_data["mask"],
            finger_landmarks,
            landmarks,
            min_area=500
        )

        if mask_pixel is not None:
            mask = mask_pixel
            method_used = "pixel-level"
            print(f"  Finger isolated using pixel-level segmentation")

            # Debug: Show ROI mask and intersection
            if debug_dir:
                h, w = image_shape
                roi_mask = _create_finger_roi_mask(finger_landmarks, landmarks, (h, w))
                save_debug_image(roi_mask, "15a_finger_roi_mask.png", debug_dir)

                # Show intersection process
                intersection = cv2.bitwise_and(hand_data["mask"], roi_mask)
                save_debug_image(intersection, "15b_roi_hand_intersection.png", debug_dir)

                # In debug mode, also generate polygon for comparison
                mask_polygon = _create_finger_mask(landmarks, indices, image_shape,
                                                   image=image, debug_dir=debug_dir)
        else:
            print(f"  Pixel-level segmentation failed, falling back to polygon")

    # Fallback to polygon-based approach
    if mask_pixel is None:
        mask_polygon = _create_finger_mask(landmarks, indices, image_shape,
                                           image=image, debug_dir=debug_dir)
        if mask_polygon is not None:
            mask = mask_polygon
            method_used = "polygon"
            print(f"  Finger isolated using polygon-based segmentation (fallback)")
        else:
            print(f"  Both segmentation methods failed")
            return None

    # Debug: Compare both methods if available
    if debug_dir and image is not None:
        # Create comparison image
        if mask_pixel is not None and mask_polygon is not None:
            comparison_img = image.copy()

            # Draw polygon contour in red
            contours_poly, _ = cv2.findContours(mask_polygon, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
            if contours_poly:
                cv2.drawContours(comparison_img, contours_poly, -1, Color.RED,
                               Size.CONTOUR_THICK, cv2.LINE_AA)

            # Draw pixel-level contour in green
            contours_pixel, _ = cv2.findContours(mask_pixel, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)
            if contours_pixel:
                cv2.drawContours(comparison_img, contours_pixel, -1, Color.GREEN,
                               Size.CONTOUR_THICK, cv2.LINE_AA)

            # Add legend
            y = Layout.TITLE_Y
            texts = [
                ("Pixel-level (accurate)", Color.GREEN),
                ("Polygon (synthetic)", Color.RED),
            ]
            for text, color in texts:
                cv2.putText(comparison_img, text, (Layout.TEXT_OFFSET_X, y), FONT_FACE,
                           FontScale.BODY, Color.BLACK, FontThickness.LABEL_OUTLINE, cv2.LINE_AA)
                cv2.putText(comparison_img, text, (Layout.TEXT_OFFSET_X, y), FONT_FACE,
                           FontScale.BODY, color, FontThickness.LABEL, cv2.LINE_AA)
                y += Layout.LINE_SPACING

            save_debug_image(comparison_img, "17a_method_comparison.png", debug_dir)

        # Overlay final mask on original
        mask_overlay = image.copy()
        mask_colored = np.zeros_like(image)
        color = Color.GREEN if method_used == "pixel-level" else Color.MAGENTA
        mask_colored[mask > 0] = color
        mask_overlay = cv2.addWeighted(mask_overlay, 0.6, mask_colored, 0.4, 0)

        # Add method label
        text = f"Method: {method_used}"
        cv2.putText(mask_overlay, text, (Layout.TEXT_OFFSET_X, Layout.TITLE_Y), FONT_FACE,
                   FontScale.BODY, Color.BLACK, FontThickness.LABEL_OUTLINE, cv2.LINE_AA)
        cv2.putText(mask_overlay, text, (Layout.TEXT_OFFSET_X, Layout.TITLE_Y), FONT_FACE,
                   FontScale.BODY, Color.WHITE, FontThickness.LABEL, cv2.LINE_AA)

        save_debug_image(mask_overlay, "18_finger_mask_overlay.png", debug_dir)

    return {
        "mask": mask,
        "landmarks": finger_landmarks,
        "base_point": finger_landmarks[0],  # MCP joint
        "tip_point": finger_landmarks[3],   # Fingertip
        "finger_name": finger,
        "method": method_used,  # Track which method was used
    }


def _create_finger_mask(
    all_landmarks: np.ndarray,
    finger_indices: List[int],
    shape: Tuple[int, int],
    width_factor: float = 2.5,
    image: Optional[np.ndarray] = None,
    debug_dir: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Create a binary mask for a single finger.

    Args:
        all_landmarks: All 21 hand landmarks
        finger_indices: Indices of the 4 finger landmarks
        shape: (height, width) of output mask
        width_factor: Multiplier for estimated finger width
        image: Optional original image for debug visualization
        debug_dir: Optional directory to save debug images

    Returns:
        Binary mask of finger region
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    finger_landmarks = all_landmarks[finger_indices]

    # Estimate finger width based on joint spacing
    # Use the distance between adjacent fingers as reference
    mcp_idx = finger_indices[0]

    # Find adjacent finger MCPs for width estimation
    adjacent_distances = []
    for other_finger, other_indices in FINGER_LANDMARKS.items():
        other_mcp = other_indices[0]
        if other_mcp != mcp_idx:
            dist = np.linalg.norm(all_landmarks[mcp_idx] - all_landmarks[other_mcp])
            adjacent_distances.append(dist)

    if adjacent_distances:
        # Finger width is approximately 1/3 to 1/2 of inter-finger distance
        estimated_width = min(adjacent_distances) * 0.4 * width_factor
    else:
        # Fallback: use finger length / 6
        finger_length = np.linalg.norm(finger_landmarks[3] - finger_landmarks[0])
        estimated_width = finger_length / 6 * width_factor

    # Create polygon along finger with estimated width
    # For each landmark, create left and right edge points
    polygon_points = []

    for i in range(len(finger_landmarks)):
        pt = finger_landmarks[i]

        # Direction along finger
        if i < len(finger_landmarks) - 1:
            direction = finger_landmarks[i + 1] - pt
        else:
            direction = pt - finger_landmarks[i - 1]

        # Perpendicular direction
        perp = np.array([-direction[1], direction[0]])
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 0:
            perp = perp / perp_norm

        # Width varies along finger (wider at base, narrower at tip)
        width_scale = 1.0 - 0.3 * (i / (len(finger_landmarks) - 1))
        half_width = estimated_width * width_scale / 2

        # Add left and right points
        left = pt + perp * half_width
        right = pt - perp * half_width
        polygon_points.append((left, right))

    # Build polygon: go up left side, then down right side
    polygon = []
    for left, right in polygon_points:
        polygon.append(left)
    for left, right in reversed(polygon_points):
        polygon.append(right)

    polygon = np.array(polygon, dtype=np.int32)

    # Debug: Visualize finger polygon construction
    if debug_dir and image is not None:
        polygon_img = image.copy()
        # Draw left and right edges
        for i, (left, right) in enumerate(polygon_points):
            cv2.circle(polygon_img, (int(left[0]), int(left[1])), 5, Color.GREEN, -1)
            cv2.circle(polygon_img, (int(right[0]), int(right[1])), 5, Color.RED, -1)
        # Draw polygon outline
        cv2.polylines(polygon_img, [polygon], True, Color.CYAN, Size.LINE_THICK)
        save_debug_image(polygon_img, "15_finger_polygon.png", debug_dir)

    # Fill the polygon
    cv2.fillPoly(mask, [polygon], 255)

    # Extend mask slightly towards palm for complete finger coverage
    # Add a region from MCP towards wrist
    mcp = finger_landmarks[0]
    wrist = all_landmarks[WRIST_LANDMARK]
    palm_direction = mcp - wrist
    palm_direction = palm_direction / (np.linalg.norm(palm_direction) + 1e-8)

    # Extend base by ~20% of finger length towards palm
    finger_length = np.linalg.norm(finger_landmarks[3] - finger_landmarks[0])
    extension = palm_direction * finger_length * 0.15
    extended_base = mcp - extension

    # Create extension polygon
    perp = np.array([-palm_direction[1], palm_direction[0]])
    half_width = estimated_width / 2
    ext_polygon = np.array([
        mcp + perp * half_width,
        mcp - perp * half_width,
        extended_base - perp * half_width * 0.8,
        extended_base + perp * half_width * 0.8,
    ], dtype=np.int32)

    # Debug: Visualize palm extension
    if debug_dir and image is not None:
        ext_img = image.copy()
        cv2.fillPoly(ext_img, [ext_polygon], Color.YELLOW)
        # Draw direction vector
        cv2.arrowedLine(ext_img, (int(mcp[0]), int(mcp[1])),
                       (int(extended_base[0]), int(extended_base[1])),
                       Color.CYAN, Size.LINE_THICK, tipLength=0.3)
        save_debug_image(ext_img, "16_palm_extension.png", debug_dir)

    cv2.fillPoly(mask, [ext_polygon], 255)

    # Debug: Save raw finger mask
    if debug_dir:
        save_debug_image(mask, "17_raw_finger_mask.png", debug_dir)

    return mask


def clean_mask(
    mask: np.ndarray,
    min_area: int = 1000,
    debug_dir: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Clean a binary mask by extracting largest component and applying morphology.

    Args:
        mask: Input binary mask
        min_area: Minimum valid area in pixels
        debug_dir: Optional directory to save debug images

    Returns:
        Cleaned binary mask, or None if no valid component found
    """
    if mask is None or mask.size == 0:
        return None

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return None

    # Find largest component (excluding background at index 0)
    largest_idx = 1
    largest_area = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_idx = i

    if largest_area < min_area:
        return None

    # Debug: Visualize connected components
    if debug_dir:
        # Create a dummy color image for visualization
        h, w = mask.shape
        comp_img = np.zeros((h, w, 3), dtype=np.uint8)
        comp_img = draw_component_stats(comp_img, labels, stats, largest_idx)
        save_debug_image(comp_img, "19_connected_components.png", debug_dir)

    # Create mask with only the largest component
    cleaned = np.zeros_like(mask)
    cleaned[labels == largest_idx] = 255

    # Debug: Save largest component
    if debug_dir:
        save_debug_image(cleaned, "20_largest_component.png", debug_dir)

    # Apply morphological smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    if debug_dir:
        save_debug_image(cleaned, "21_morph_close.png", debug_dir)

    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    if debug_dir:
        save_debug_image(cleaned, "22_morph_open.png", debug_dir)

    # Smooth edges with Gaussian blur and re-threshold
    cleaned = cv2.GaussianBlur(cleaned, (5, 5), 0)
    _, cleaned = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)

    if debug_dir:
        save_debug_image(cleaned, "23_gaussian_blur.png", debug_dir)
        save_debug_image(cleaned, "24_final_cleaned_mask.png", debug_dir)

    return cleaned


def get_finger_contour(
    mask: np.ndarray,
    smooth: bool = True,
    debug_dir: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Extract outer contour from finger mask.

    Args:
        mask: Binary finger mask
        smooth: Whether to apply contour smoothing
        debug_dir: Optional directory to save debug images (currently unused, contours shown in main debug overlay)

    Returns:
        Contour points as Nx2 array, or None if no contour found
    """
    if mask is None:
        return None

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Reshape to Nx2
    contour = largest_contour.reshape(-1, 2)

    if smooth and len(contour) > 10:
        # Apply contour smoothing using approximation
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        smoothed = cv2.approxPolyDP(largest_contour, epsilon, True)
        contour = smoothed.reshape(-1, 2)

    return contour.astype(np.float32)
