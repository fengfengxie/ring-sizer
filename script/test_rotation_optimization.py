#!/usr/bin/env python3
"""
Test script for image rotation optimization.
Tests that measurements are consistent across different hand orientations.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.finger_segmentation import detect_hand_orientation, normalize_hand_orientation, segment_hand


def test_orientation_detection():
    """Test orientation detection with synthetic landmarks."""
    print("=" * 60)
    print("Test 1: Orientation Detection")
    print("=" * 60)
    
    # Test case 1: Fingers pointing up (canonical) - 0°
    landmarks_up = np.array([[0.5, 0.8], [0.5, 0.2]])  # wrist at 0, middle_tip at 1
    # Extend with dummy landmarks
    full_landmarks = np.zeros((21, 2))
    full_landmarks[0] = landmarks_up[0]  # wrist
    full_landmarks[12] = landmarks_up[1]  # middle tip
    
    rotation = detect_hand_orientation(full_landmarks)
    print(f"Fingers pointing up → rotation needed: {rotation}° (expected: 0°)")
    assert rotation == 0, f"Expected 0°, got {rotation}°"
    
    # Test case 2: Fingers pointing right - 90°
    landmarks_right = np.array([[0.2, 0.5], [0.8, 0.5]])  # wrist left, tip right
    full_landmarks[0] = landmarks_right[0]
    full_landmarks[12] = landmarks_right[1]
    
    rotation = detect_hand_orientation(full_landmarks)
    print(f"Fingers pointing right → rotation needed: {rotation}° (expected: 270°)")
    assert rotation == 270, f"Expected 270°, got {rotation}°"
    
    # Test case 3: Fingers pointing down - 180°
    landmarks_down = np.array([[0.5, 0.2], [0.5, 0.8]])  # wrist top, tip bottom
    full_landmarks[0] = landmarks_down[0]
    full_landmarks[12] = landmarks_down[1]
    
    rotation = detect_hand_orientation(full_landmarks)
    print(f"Fingers pointing down → rotation needed: {rotation}° (expected: 180°)")
    assert rotation == 180, f"Expected 180°, got {rotation}°"
    
    # Test case 4: Fingers pointing left - 270°
    landmarks_left = np.array([[0.8, 0.5], [0.2, 0.5]])  # wrist right, tip left
    full_landmarks[0] = landmarks_left[0]
    full_landmarks[12] = landmarks_left[1]
    
    rotation = detect_hand_orientation(full_landmarks)
    print(f"Fingers pointing left → rotation needed: {rotation}° (expected: 90°)")
    assert rotation == 90, f"Expected 90°, got {rotation}°"
    
    print("✓ All orientation detection tests passed!\n")


def test_image_rotation():
    """Test actual image rotation."""
    print("=" * 60)
    print("Test 2: Image Rotation")
    print("=" * 60)
    
    # Create a test image with asymmetric pattern
    img = np.zeros((400, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (100, 100), (255, 0, 0), -1)  # Blue square top-left
    cv2.rectangle(img, (200, 300), (290, 390), (0, 255, 0), -1)  # Green square bottom-right
    cv2.putText(img, "UP", (130, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Test landmarks (fingers pointing up)
    landmarks = np.zeros((21, 2))
    landmarks[0] = [0.5, 0.8]  # wrist
    landmarks[12] = [0.5, 0.2]  # middle tip
    
    # No rotation needed
    rotated, angle = normalize_hand_orientation(img, landmarks, None)
    print(f"Original image (fingers up) → rotation: {angle}°")
    assert angle == 0, f"Expected 0°, got {angle}°"
    assert rotated.shape == img.shape, "Image shape changed unexpectedly"
    
    # Test with 90° rotated landmarks
    landmarks[0] = [0.2, 0.5]  # wrist left
    landmarks[12] = [0.8, 0.5]  # tip right
    rotated, angle = normalize_hand_orientation(img, landmarks, None)
    print(f"Fingers pointing right → rotation: {angle}°")
    assert angle == 270, f"Expected 270°, got {angle}°"
    # After 270° CW rotation (= 90° CCW), width and height should swap
    assert rotated.shape[0] == img.shape[1], "Height should be original width"
    assert rotated.shape[1] == img.shape[0], "Width should be original height"
    
    print("✓ All image rotation tests passed!\n")


def test_with_real_image():
    """Test with real image if available."""
    print("=" * 60)
    print("Test 3: Real Image Test")
    print("=" * 60)
    
    test_image_path = Path("input/test_sample2.jpg")
    if not test_image_path.exists():
        print("Skipping real image test (input/test_sample2.jpg not found)")
        return
    
    image = cv2.imread(str(test_image_path))
    if image is None:
        print(f"Could not load image: {test_image_path}")
        return
    
    print(f"Loaded image: {test_image_path}")
    print(f"Original size: {image.shape[1]}x{image.shape[0]}")
    
    # Run segmentation with orientation normalization
    hand_data = segment_hand(image, debug_dir=None)
    
    if hand_data is None:
        print("⚠ No hand detected in test image")
        return
    
    print(f"✓ Hand detected: {hand_data['handedness']}")
    print(f"  Confidence: {hand_data['confidence']:.3f}")
    print(f"  Detection rotation: {hand_data.get('rotation_applied', 'N/A')}")
    print(f"  Orientation rotation: {hand_data.get('orientation_rotation', 'N/A')}°")
    
    if "canonical_image" in hand_data:
        canonical = hand_data["canonical_image"]
        print(f"  Canonical size: {canonical.shape[1]}x{canonical.shape[0]}")
    
    print("✓ Real image test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("IMAGE ROTATION OPTIMIZATION TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_orientation_detection()
        test_image_rotation()
        test_with_real_image()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
