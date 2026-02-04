#!/usr/bin/env python3
"""
Test script for Sobel edge refinement (Phase 2).

This validates the edge detection pipeline and compares with contour method.
"""

import cv2
import numpy as np
from src.card_detection import detect_credit_card, compute_scale_factor
from src.finger_segmentation import segment_hand, isolate_finger, clean_mask, get_finger_contour
from src.geometry import estimate_finger_axis, localize_ring_zone, compute_cross_section_width
from src.edge_refinement import refine_edges_sobel
import sys


def test_edge_refinement(image_path: str):
    """Test Sobel edge refinement and compare with contour method."""
    print(f"\n{'='*70}")
    print(f"Sobel Edge Refinement Test (Phase 2)")
    print(f"{'='*70}\n")

    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")

    # Detect card for scale
    print("\n" + "="*70)
    print("Card Detection & Scale Calibration")
    print("="*70)
    card_result = detect_credit_card(image)
    if card_result is None:
        print("Error: Card not detected")
        return

    px_per_cm, scale_confidence = compute_scale_factor(card_result["corners"])
    print(f"Scale: {px_per_cm:.2f} px/cm (confidence={scale_confidence:.2f})")
    print(f"Card: {card_result['width_px']:.0f}x{card_result['height_px']:.0f}px, aspect={card_result['aspect_ratio']:.3f}")

    # Segment hand and isolate finger
    print("\n" + "="*70)
    print("Hand & Finger Segmentation")
    print("="*70)
    hand_data = segment_hand(image)
    if hand_data is None:
        print("Error: No hand detected")
        return

    finger_data = isolate_finger(hand_data, finger="auto", image_shape=(h, w), image=image)
    if finger_data is None:
        print("Error: Could not isolate finger")
        return

    print(f"Finger: {finger_data['finger_name']}")

    # Clean mask and extract contour
    mask = clean_mask(finger_data["mask"])
    if mask is None:
        print("Error: Could not clean mask")
        return

    contour = get_finger_contour(mask)
    if contour is None:
        print("Error: Could not extract contour")
        return

    print(f"Contour extracted: {len(contour)} points")

    # Estimate axis and localize ring zone
    landmarks = finger_data["landmarks"]
    axis_data = estimate_finger_axis(mask, landmarks, method="auto")
    zone_data = localize_ring_zone(axis_data)

    print(f"Finger axis: length={axis_data['length']:.1f}px, method={axis_data['method']}")
    print(f"Ring zone: {zone_data['start_pct']*100:.0f}%-{zone_data['end_pct']*100:.0f}% from palm")

    # Method 1: Contour-based measurement (v0)
    print("\n" + "="*70)
    print("Method 1: Contour-Based Width Measurement (v0)")
    print("="*70)

    try:
        contour_result = compute_cross_section_width(
            contour, axis_data, zone_data, num_samples=20
        )
        contour_width_px = contour_result["median_width_px"]
        contour_width_cm = contour_width_px / px_per_cm
        contour_std_px = contour_result["std_width_px"]

        print(f"Median width: {contour_width_cm:.3f} cm ({contour_width_px:.1f} px)")
        print(f"Std deviation: {contour_std_px:.3f} px")
        print(f"Samples: {contour_result['num_samples']}/20")
        print(f"Method: Contour intersection")

    except Exception as e:
        print(f"Error: Contour measurement failed: {e}")
        contour_result = None
        contour_width_cm = None

    # Method 2: Sobel edge refinement (v1)
    print("\n" + "="*70)
    print("Method 2: Sobel Edge Refinement (v1)")
    print("="*70)

    try:
        # Don't use expected width for initial test
        # Let Sobel find edges without strict validation

        sobel_result = refine_edges_sobel(
            image=image,
            axis_data=axis_data,
            zone_data=zone_data,
            scale_px_per_cm=px_per_cm,
            finger_mask=mask,  # Use mask to constrain edge search
            sobel_threshold=15.0,  # Lower threshold for better detection
            kernel_size=3,
            rotate_align=False,
            expected_width_px=None,  # No strict validation
        )

        sobel_width_cm = sobel_result["median_width_cm"]
        sobel_width_px = sobel_result["median_width_px"]
        sobel_std_px = sobel_result["std_width_px"]

        print(f"Median width: {sobel_width_cm:.3f} cm ({sobel_width_px:.1f} px)")
        print(f"Std deviation: {sobel_std_px:.3f} px")
        print(f"Samples: {sobel_result['num_samples']}")
        print(f"Success rate: {sobel_result['edge_detection_success_rate']*100:.1f}%")
        print(f"Outliers removed: {sobel_result['outliers_removed']}")
        print(f"Method: Sobel gradient edge detection")

        # ROI info
        roi_data = sobel_result["roi_data"]
        print(f"\nROI details:")
        print(f"  Size: {roi_data['roi_width']}x{roi_data['roi_height']} px")
        print(f"  Rotation: {roi_data['rotation_angle']:.1f}°")

        # Gradient info
        gradient_data = sobel_result["gradient_data"]
        print(f"\nGradient details:")
        print(f"  Kernel size: {gradient_data['kernel_size']}")
        print(f"  Filter orientation: {gradient_data['filter_orientation']}")

        # Edge detection info
        edge_data = sobel_result["edge_data"]
        print(f"\nEdge detection details:")
        print(f"  Valid rows: {edge_data['num_valid_rows']}/{len(edge_data['valid_rows'])}")
        print(f"  Avg left edge strength: {np.mean(edge_data['edge_strengths_left'][edge_data['valid_rows']]):.1f}")
        print(f"  Avg right edge strength: {np.mean(edge_data['edge_strengths_right'][edge_data['valid_rows']]):.1f}")

    except Exception as e:
        print(f"Error: Sobel edge refinement failed: {e}")
        import traceback
        traceback.print_exc()
        sobel_result = None
        sobel_width_cm = None

    # Comparison
    if contour_result and sobel_result:
        print("\n" + "="*70)
        print("Method Comparison")
        print("="*70)

        diff_cm = sobel_width_cm - contour_width_cm
        diff_pct = (diff_cm / contour_width_cm) * 100

        print(f"Contour method: {contour_width_cm:.3f} cm")
        print(f"Sobel method:   {sobel_width_cm:.3f} cm")
        print(f"Difference:     {diff_cm:+.3f} cm ({diff_pct:+.2f}%)")
        print(f"Std dev ratio:  {sobel_std_px/contour_std_px:.2f}x")

        if abs(diff_pct) < 2:
            print(f"\n✓ Methods agree closely (<2% difference)")
        elif abs(diff_pct) < 5:
            print(f"\n⚠ Methods differ moderately (2-5% difference)")
        else:
            print(f"\n✗ Methods differ significantly (>5% difference)")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "input/test_sample2.jpg"

    test_edge_refinement(image_path)
