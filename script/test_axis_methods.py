#!/usr/bin/env python3
"""
Test script to compare landmark-based vs PCA-based axis estimation.

This demonstrates the Phase 1 implementation.
"""

import cv2
import numpy as np
from src.finger_segmentation import segment_hand, isolate_finger
from src.geometry import estimate_finger_axis
import sys


def test_axis_comparison(image_path: str):
    """Compare landmark-based and PCA-based axis estimation."""
    print(f"\n{'='*70}")
    print(f"Axis Estimation Comparison Test")
    print(f"{'='*70}\n")

    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")

    # Segment hand
    print("\nSegmenting hand...")
    hand_data = segment_hand(image)
    if hand_data is None:
        print("Error: No hand detected")
        return

    print(f"Hand detected: {hand_data['handedness']}, confidence={hand_data['confidence']:.2f}")

    # Isolate finger
    print("\nIsolating finger...")
    finger_data = isolate_finger(hand_data, finger="auto", image_shape=(h, w), image=image)
    if finger_data is None:
        print("Error: Could not isolate finger")
        return

    print(f"Finger isolated: {finger_data['finger_name']}")

    mask = finger_data["mask"]
    landmarks = finger_data["landmarks"]

    print(f"\nLandmark positions:")
    landmark_names = ["MCP (knuckle)", "PIP joint", "DIP joint", "TIP (fingertip)"]
    for i, (name, pos) in enumerate(zip(landmark_names, landmarks)):
        print(f"  {name}: ({pos[0]:.1f}, {pos[1]:.1f})")

    # Test landmark-based axis (3 methods)
    print(f"\n{'='*70}")
    print(f"Landmark-Based Axis Estimation")
    print(f"{'='*70}\n")

    for method in ["endpoints", "linear_fit", "median_direction"]:
        axis_data = estimate_finger_axis(mask, landmarks, method="landmarks", landmark_method=method)
        print(f"Method: {method}")
        print(f"  Length: {axis_data['length']:.1f} px")
        print(f"  Center: ({axis_data['center'][0]:.1f}, {axis_data['center'][1]:.1f})")
        print(f"  Direction: ({axis_data['direction'][0]:.3f}, {axis_data['direction'][1]:.3f})")
        print(f"  Method used: {axis_data['method']}")
        print()

    # Test PCA-based axis
    print(f"{'='*70}")
    print(f"PCA-Based Axis Estimation (v0 method)")
    print(f"{'='*70}\n")

    axis_data_pca = estimate_finger_axis(mask, landmarks, method="pca")
    print(f"  Length: {axis_data_pca['length']:.1f} px")
    print(f"  Center: ({axis_data_pca['center'][0]:.1f}, {axis_data_pca['center'][1]:.1f})")
    print(f"  Direction: ({axis_data_pca['direction'][0]:.3f}, {axis_data_pca['direction'][1]:.3f})")
    print(f"  Method used: {axis_data_pca['method']}")
    print()

    # Test auto mode
    print(f"{'='*70}")
    print(f"Auto Mode (defaults to landmark if available)")
    print(f"{'='*70}\n")

    axis_data_auto = estimate_finger_axis(mask, landmarks, method="auto")
    print(f"  Length: {axis_data_auto['length']:.1f} px")
    print(f"  Center: ({axis_data_auto['center'][0]:.1f}, {axis_data_auto['center'][1]:.1f})")
    print(f"  Direction: ({axis_data_auto['direction'][0]:.3f}, {axis_data_auto['direction'][1]:.3f})")
    print(f"  Method used: {axis_data_auto['method']}")
    print()

    # Compare linear_fit landmark with PCA
    axis_landmark = estimate_finger_axis(mask, landmarks, method="landmarks", landmark_method="linear_fit")

    print(f"{'='*70}")
    print(f"Comparison: Landmark (linear_fit) vs PCA")
    print(f"{'='*70}\n")

    length_diff = axis_landmark['length'] - axis_data_pca['length']
    center_diff = np.linalg.norm(axis_landmark['center'] - axis_data_pca['center'])

    # Calculate angle between directions
    dot_product = np.dot(axis_landmark['direction'], axis_data_pca['direction'])
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    print(f"Length difference: {length_diff:+.1f} px ({abs(length_diff/axis_data_pca['length']*100):.2f}%)")
    print(f"Center displacement: {center_diff:.1f} px")
    print(f"Direction angle difference: {angle_deg:.2f}°")

    if angle_deg < 5:
        print(f"\n✓ Methods agree closely (angle < 5°)")
    elif angle_deg < 15:
        print(f"\n⚠ Methods differ moderately (5° < angle < 15°)")
    else:
        print(f"\n✗ Methods differ significantly (angle > 15°)")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "input/test_sample2.jpg"

    test_axis_comparison(image_path)
