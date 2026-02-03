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
) -> Optional[Dict[str, Any]]:
    """
    Detect and segment hand from image using MediaPipe.

    Args:
        image: Input BGR image
        max_dimension: Maximum dimension for processing (large images are resized)

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

    # Generate hand mask at original resolution
    mask = _create_hand_mask(landmarks, (h, w))

    return {
        "landmarks": landmarks,
        "landmarks_normalized": landmarks_normalized,
        "mask": mask,
        "confidence": handedness[0].score,
        "handedness": handedness[0].category_name,
        "rotation_applied": rotation_code,
    }


def _create_hand_mask(landmarks: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a binary mask of the hand region from landmarks.

    Args:
        landmarks: 21x2 array of landmark pixel coordinates
        shape: (height, width) of output mask

    Returns:
        Binary mask (uint8, 0 or 255)
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Create convex hull of all landmarks
    hull_points = cv2.convexHull(landmarks.astype(np.int32))
    cv2.fillConvexPoly(mask, hull_points, 255)

    # Also fill individual finger regions for better coverage
    for finger_name, indices in FINGER_LANDMARKS.items():
        finger_pts = landmarks[indices].astype(np.int32)
        # Create a polygon along the finger
        cv2.fillConvexPoly(mask, finger_pts, 255)

    # Fill thumb
    thumb_pts = landmarks[THUMB_LANDMARKS].astype(np.int32)
    cv2.fillConvexPoly(mask, thumb_pts, 255)

    # Apply morphological operations to smooth the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

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


def isolate_finger(
    hand_data: Dict[str, Any],
    finger: FingerIndex = "auto",
    image_shape: Optional[Tuple[int, int]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Isolate a specific finger from hand segmentation data.

    Args:
        hand_data: Output from segment_hand()
        finger: Which finger to isolate, or "auto" to select most extended
        image_shape: (height, width) for mask generation

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
    if finger == "auto":
        # Find the most extended finger
        best_finger = None
        best_score = -1

        for finger_name, indices in FINGER_LANDMARKS.items():
            score = _calculate_finger_extension(landmarks, indices)
            if score > best_score:
                best_score = score
                best_finger = finger_name

        if best_finger is None:
            return None
        finger = best_finger

    if finger not in FINGER_LANDMARKS:
        return None

    indices = FINGER_LANDMARKS[finger]
    finger_landmarks = landmarks[indices]

    # Create finger mask
    mask = _create_finger_mask(landmarks, indices, image_shape)

    if mask is None:
        return None

    return {
        "mask": mask,
        "landmarks": finger_landmarks,
        "base_point": finger_landmarks[0],  # MCP joint
        "tip_point": finger_landmarks[3],   # Fingertip
        "finger_name": finger,
    }


def _create_finger_mask(
    all_landmarks: np.ndarray,
    finger_indices: List[int],
    shape: Tuple[int, int],
    width_factor: float = 2.5,
) -> Optional[np.ndarray]:
    """
    Create a binary mask for a single finger.

    Args:
        all_landmarks: All 21 hand landmarks
        finger_indices: Indices of the 4 finger landmarks
        shape: (height, width) of output mask
        width_factor: Multiplier for estimated finger width

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
    cv2.fillPoly(mask, [ext_polygon], 255)

    return mask


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

    # Create mask with only the largest component
    cleaned = np.zeros_like(mask)
    cleaned[labels == largest_idx] = 255

    # Apply morphological smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Smooth edges with Gaussian blur and re-threshold
    cleaned = cv2.GaussianBlur(cleaned, (5, 5), 0)
    _, cleaned = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)

    return cleaned


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
