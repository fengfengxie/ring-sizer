"""
Debug visualization observer for the ring measurement pipeline.

This module provides a non-intrusive way to capture and visualize intermediate
processing stages without polluting core algorithm implementations.

It also contains all drawing utility functions used for debug visualizations.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any, Callable, List, Tuple
from pathlib import Path

# Import visualization constants
from src.viz_constants import (
    FONT_FACE, FontScale, FontThickness, Color, Size, Layout
)


class DebugObserver:
    """
    Observer for capturing and saving intermediate processing stages.
    
    This class provides methods to save images and visualizations during
    algorithm execution without requiring core functions to handle I/O directly.
    """
    
    def __init__(self, debug_dir: str):
        """
        Initialize debug observer.
        
        Args:
            debug_dir: Directory where debug images will be saved
        """
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._stage_counter = {}
    
    def save_stage(self, name: str, image: np.ndarray) -> None:
        """
        Save an intermediate processing stage image.
        
        Args:
            name: Stage name (used as filename prefix)
            image: Image to save
        """
        if image is None or image.size == 0:
            return
        
        # Add counter for stages with multiple saves
        if name in self._stage_counter:
            self._stage_counter[name] += 1
            filename = f"{name}_{self._stage_counter[name]}.png"
        else:
            self._stage_counter[name] = 0
            filename = f"{name}.png"
        
        self._save_with_compression(image, filename)
    
    def draw_and_save(self, name: str, image: np.ndarray,
                      draw_func: Callable, *args, **kwargs) -> None:
        """
        Apply a drawing function to an image and save the result.
        
        Args:
            name: Stage name for the output file
            image: Base image to draw on
            draw_func: Function that takes (image, *args, **kwargs) and returns annotated image
            *args, **kwargs: Arguments to pass to draw_func
        """
        if image is None or image.size == 0:
            return
        
        annotated = draw_func(image, *args, **kwargs)
        self.save_stage(name, annotated)
    
    def _save_with_compression(self, image: np.ndarray, filename: str) -> None:
        """
        Save image with compression and optional downsampling.
        
        Args:
            image: Image to save
            filename: Output filename
        """
        output_path = self.debug_dir / filename
        
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


# Backward compatibility helper
def save_debug_image(image: np.ndarray, filename: str, debug_dir: Optional[str]) -> None:
    """
    Legacy function for saving debug images.
    
    This function is kept for backward compatibility during migration.
    New code should use DebugObserver directly.
    
    Args:
        image: Image to save
        filename: Output filename
        debug_dir: Directory to save to (if None, skip saving)
    """
    if debug_dir is None:
        return
    
    observer = DebugObserver(debug_dir)
    observer._save_with_compression(image, filename)


# =============================================================================
# Drawing Functions for Debug Visualization
# =============================================================================

# Hand landmark and finger constants (from finger_segmentation.py)
FINGER_LANDMARKS = {
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

THUMB_LANDMARKS = [1, 2, 3, 4]

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

FINGER_COLORS = {
    "thumb": Color.RED,
    "index": Color.CYAN,
    "middle": Color.YELLOW,
    "ring": Color.MAGENTA,
    "pinky": Color.ORANGE,
}


# --- Finger Segmentation Drawing Functions ---

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


# --- Card Detection Drawing Functions ---

def draw_contours_overlay(
    image: np.ndarray,
    contours: List[np.ndarray],
    title: str,
    color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Draw contours on an image overlay.

    Args:
        image: Original image
        contours: List of contours to draw
        title: Title for the visualization
        color: BGR color for contours (default: Color.GREEN)

    Returns:
        Annotated image
    """
    if color is None:
        color = Color.GREEN

    overlay = image.copy()

    # Draw all contours
    for contour in contours:
        if len(contour) == 4:
            # Draw quadrilateral
            pts = contour.reshape(4, 2).astype(np.int32)
            cv2.polylines(overlay, [pts], True, color, Size.CONTOUR_NORMAL)

    # Add title with outline for visibility
    cv2.putText(
        overlay, title, (Layout.TEXT_OFFSET_X, Layout.TITLE_Y),
        FONT_FACE, FontScale.TITLE, Color.WHITE,
        FontThickness.TITLE_OUTLINE, cv2.LINE_AA
    )
    cv2.putText(
        overlay, title, (Layout.TEXT_OFFSET_X, Layout.TITLE_Y),
        FONT_FACE, FontScale.TITLE, color,
        FontThickness.TITLE, cv2.LINE_AA
    )

    # Add count with outline
    count_text = f"Candidates: {len(contours)}"
    cv2.putText(
        overlay, count_text, (Layout.TEXT_OFFSET_X, Layout.SUBTITLE_Y),
        FONT_FACE, FontScale.SUBTITLE, Color.WHITE,
        FontThickness.SUBTITLE_OUTLINE, cv2.LINE_AA
    )
    cv2.putText(
        overlay, count_text, (Layout.TEXT_OFFSET_X, Layout.SUBTITLE_Y),
        FONT_FACE, FontScale.SUBTITLE, color,
        FontThickness.SUBTITLE, cv2.LINE_AA
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
        Color.GREEN,    # Green - best
        Color.YELLOW,   # Yellow
        Color.ORANGE,   # Orange
        Color.MAGENTA,  # Magenta
        Color.PINK      # Pink
    ]

    for idx, (corners, score, details) in enumerate(candidates):
        color = colors[idx % len(colors)]

        # Draw quadrilateral
        pts = corners.reshape(4, 2).astype(np.int32)
        cv2.polylines(overlay, [pts], True, color, Size.CONTOUR_NORMAL)

        # Draw corner circles
        for pt in pts:
            cv2.circle(overlay, tuple(pt), Size.CORNER_RADIUS, color, -1)

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
            FONT_FACE, FontScale.LABEL, Color.BLACK,
            FontThickness.LABEL_OUTLINE, cv2.LINE_AA
        )
        cv2.putText(
            overlay, text, text_pos,
            FONT_FACE, FontScale.LABEL, color,
            FontThickness.LABEL, cv2.LINE_AA
        )

    # Add title with outline
    cv2.putText(
        overlay, title, (Layout.TEXT_OFFSET_X, Layout.TITLE_Y),
        FONT_FACE, FontScale.TITLE, Color.WHITE,
        FontThickness.TITLE_OUTLINE, cv2.LINE_AA
    )
    cv2.putText(
        overlay, title, (Layout.TEXT_OFFSET_X, Layout.TITLE_Y),
        FONT_FACE, FontScale.TITLE, Color.CYAN,
        FontThickness.TITLE, cv2.LINE_AA
    )

    return overlay
