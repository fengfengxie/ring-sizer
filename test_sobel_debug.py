#!/usr/bin/env python3
"""
Debug Sobel edge detection to understand why success rate is low.
"""

import cv2
import numpy as np
from src.card_detection import detect_credit_card, compute_scale_factor
from src.finger_segmentation import segment_hand, isolate_finger, clean_mask
from src.geometry import estimate_finger_axis, localize_ring_zone
from src.edge_refinement import extract_ring_zone_roi, apply_sobel_filters, detect_edges_per_row


image_path = "input/test_sample2.jpg"
image = cv2.imread(image_path)
h, w = image.shape[:2]

# Get scale
card_result = detect_credit_card(image)
px_per_cm, _ = compute_scale_factor(card_result["corners"])

# Get finger
hand_data = segment_hand(image)
finger_data = isolate_finger(hand_data, finger="auto", image_shape=(h, w), image=image)
mask = clean_mask(finger_data["mask"])
landmarks = finger_data["landmarks"]

# Get axis and zone
axis_data = estimate_finger_axis(mask, landmarks, method="auto")
zone_data = localize_ring_zone(axis_data)

# Extract ROI
roi_data = extract_ring_zone_roi(image, axis_data, zone_data, padding=50)

print(f"ROI size: {roi_data['roi_width']}x{roi_data['roi_height']}")

# Apply Sobel with different thresholds
for threshold in [10, 20, 30, 40, 50]:
    gradient_data = apply_sobel_filters(roi_data["roi_image"], kernel_size=3)

    # Analyze gradient magnitude
    grad_mag = gradient_data["gradient_magnitude"]
    print(f"\nThreshold: {threshold}")
    print(f"  Gradient magnitude: min={grad_mag.min():.1f}, max={grad_mag.max():.1f}, mean={grad_mag.mean():.1f}")
    print(f"  Pixels above threshold: {np.sum(grad_mag > threshold)}/{grad_mag.size} ({np.sum(grad_mag > threshold)/grad_mag.size*100:.1f}%)")

    # Test edge detection
    edge_data = detect_edges_per_row(gradient_data, roi_data, threshold=threshold, expected_width_px=None)
    success_rate = edge_data["num_valid_rows"] / len(edge_data["valid_rows"])
    print(f"  Edge detection success: {edge_data['num_valid_rows']}/{len(edge_data['valid_rows'])} rows ({success_rate*100:.1f}%)")

    if edge_data["num_valid_rows"] > 0:
        valid_widths = edge_data["right_edges"][edge_data["valid_rows"]] - edge_data["left_edges"][edge_data["valid_rows"]]
        print(f"  Width range: {valid_widths.min():.1f}-{valid_widths.max():.1f}px, median={np.median(valid_widths):.1f}px")

# Save ROI and gradient images for inspection
cv2.imwrite("output/debug_roi.png", roi_data["roi_image"])
gradient_data = apply_sobel_filters(roi_data["roi_image"], kernel_size=3)
cv2.imwrite("output/debug_gradient_mag.png", gradient_data["gradient_mag_normalized"])
cv2.imwrite("output/debug_gradient_x.png", gradient_data["gradient_x_normalized"])
cv2.imwrite("output/debug_gradient_y.png", gradient_data["gradient_y_normalized"])

print(f"\nDebug images saved to output/debug_*.png")
