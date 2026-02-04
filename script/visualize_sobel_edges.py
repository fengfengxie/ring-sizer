#!/usr/bin/env python3
"""
Visualize Sobel edge detection to understand what edges are being found.
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

# Setup
card_result = detect_credit_card(image)
px_per_cm, _ = compute_scale_factor(card_result["corners"])

hand_data = segment_hand(image)
finger_data = isolate_finger(hand_data, finger="auto", image_shape=(h, w), image=image)
mask = clean_mask(finger_data["mask"])
landmarks = finger_data["landmarks"]

axis_data = estimate_finger_axis(mask, landmarks, method="auto")
zone_data = localize_ring_zone(axis_data)

# Extract ROI and apply Sobel
roi_data = extract_ring_zone_roi(image, axis_data, zone_data, padding=50)
gradient_data = apply_sobel_filters(roi_data["roi_image"], kernel_size=3)
edge_data = detect_edges_per_row(gradient_data, roi_data, threshold=15.0, expected_width_px=None)

# Create visualization
roi_vis = cv2.cvtColor(roi_data["roi_image"], cv2.COLOR_GRAY2BGR)

# Draw detected edges
for row in range(len(edge_data["valid_rows"])):
    if edge_data["valid_rows"][row]:
        left_x = int(edge_data["left_edges"][row])
        right_x = int(edge_data["right_edges"][row])

        # Draw edge points
        cv2.circle(roi_vis, (left_x, row), 2, (0, 255, 0), -1)  # Green for left
        cv2.circle(roi_vis, (right_x, row), 2, (0, 0, 255), -1)  # Red for right

        # Draw width line
        cv2.line(roi_vis, (left_x, row), (right_x, row), (255, 255, 0), 1)  # Cyan for width

# Draw zone boundaries
zone_start_roi = roi_data["zone_start_in_roi"]
zone_end_roi = roi_data["zone_end_in_roi"]
cv2.circle(roi_vis, (int(zone_start_roi[0]), int(zone_start_roi[1])), 5, (255, 0, 255), -1)
cv2.circle(roi_vis, (int(zone_end_roi[0]), int(zone_end_roi[1])), 5, (255, 0, 255), -1)

# Add text overlay
valid_widths = edge_data["right_edges"][edge_data["valid_rows"]] - edge_data["left_edges"][edge_data["valid_rows"]]
median_width_px = np.median(valid_widths)
median_width_cm = median_width_px / px_per_cm

cv2.putText(roi_vis, f"Median width: {median_width_cm:.2f}cm ({median_width_px:.0f}px)",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(roi_vis, f"Success: {edge_data['num_valid_rows']}/{len(edge_data['valid_rows'])} rows",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Save visualization
cv2.imwrite("output/sobel_edges_visualization.png", roi_vis)

# Also save gradient magnitude with edges overlaid
grad_mag_vis = cv2.cvtColor(gradient_data["gradient_mag_normalized"], cv2.COLOR_GRAY2BGR)
for row in range(len(edge_data["valid_rows"])):
    if edge_data["valid_rows"][row]:
        left_x = int(edge_data["left_edges"][row])
        right_x = int(edge_data["right_edges"][row])
        cv2.circle(grad_mag_vis, (left_x, row), 2, (0, 255, 0), -1)
        cv2.circle(grad_mag_vis, (right_x, row), 2, (0, 0, 255), -1)

cv2.imwrite("output/gradient_with_edges.png", grad_mag_vis)

print(f"Visualization saved to:")
print(f"  output/sobel_edges_visualization.png")
print(f"  output/gradient_with_edges.png")
print(f"\nMedian width: {median_width_cm:.3f}cm ({median_width_px:.1f}px)")
print(f"Success rate: {edge_data['num_valid_rows']}/{len(edge_data['valid_rows'])} ({edge_data['num_valid_rows']/len(edge_data['valid_rows'])*100:.1f}%)")
