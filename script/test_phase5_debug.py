#!/usr/bin/env python3
"""
Test script for Phase 5: Edge Refinement Debug Visualization.

Tests the complete 12-image debug pipeline for edge refinement.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.card_detection import detect_credit_card, compute_scale_factor
from src.finger_segmentation import segment_hand, isolate_finger, clean_mask, get_finger_contour
from src.geometry import estimate_finger_axis, localize_ring_zone, compute_cross_section_width
from src.edge_refinement import refine_edges_sobel


def test_edge_refinement_debug(image_path: str):
    """Test edge refinement with debug visualization."""
    
    print("=" * 80)
    print("PHASE 5: EDGE REFINEMENT DEBUG VISUALIZATION TEST")
    print("=" * 80)
    
    # Load image
    print(f"\n1. Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"✗ Failed to load image: {image_path}")
        return False
    print(f"✓ Image loaded: {image.shape}")
    
    # Detect card (Phase 2)
    print("\n2. Detecting credit card...")
    card_result = detect_credit_card(image)
    if card_result is None or not card_result.get("success", False):
        print(f"✗ Card detection failed")
        return False
    scale_result = compute_scale_factor(card_result)
    px_per_cm = scale_result["px_per_cm"]
    print(f"✓ Card detected, scale: {px_per_cm:.2f} px/cm")
    
    # Segment hand (Phase 4)
    print("\n3. Segmenting hand...")
    hand_result = segment_hand(image)
    if not hand_result["hand_detected"]:
        print(f"✗ Hand detection failed")
        return False
    print(f"✓ Hand detected, confidence: {hand_result['confidence']:.3f}")
    
    # Isolate finger
    print("\n4. Isolating finger...")
    finger_result = isolate_finger(
        hand_result,
        finger_index="auto",
        image=image
    )
    if finger_result is None:
        print(f"✗ Finger isolation failed")
        return False
    print(f"✓ Finger isolated: {finger_result['finger_index']}")
    
    # Clean mask
    print("\n5. Cleaning finger mask...")
    cleaned_mask = clean_mask(finger_result["finger_mask"])
    
    # Get contour
    print("\n6. Extracting finger contour...")
    contour_result = get_finger_contour(cleaned_mask)
    if contour_result is None:
        print(f"✗ Contour extraction failed")
        return False
    finger_contour = contour_result["contour"]
    print(f"✓ Contour extracted: {len(finger_contour)} points")
    
    # Estimate axis (v1 landmark-based)
    print("\n7. Estimating finger axis (landmark-based)...")
    axis_data = estimate_finger_axis(
        cleaned_mask,
        finger_contour,
        landmarks=finger_result.get("landmarks"),
        method="auto"  # Prefers landmarks over PCA
    )
    print(f"✓ Axis estimated: method={axis_data['method']}, length={axis_data['length']:.1f}px")
    
    # Localize ring zone
    print("\n8. Localizing ring-wearing zone...")
    zone_data = localize_ring_zone(axis_data, cleaned_mask)
    print(f"✓ Ring zone localized: length={zone_data['length']:.1f}px")
    
    # Create debug directory
    debug_dir = "output/edge_refinement_debug"
    os.makedirs(debug_dir, exist_ok=True)
    print(f"\n9. Debug directory: {debug_dir}")
    
    # Run edge refinement with debug
    print("\n10. Running Sobel edge refinement with debug visualization...")
    sobel_result = refine_edges_sobel(
        image=image,
        axis_data=axis_data,
        zone_data=zone_data,
        scale_px_per_cm=px_per_cm,
        finger_mask=cleaned_mask,
        finger_landmarks=finger_result.get("landmarks"),
        sobel_threshold=15.0,
        kernel_size=3,
        debug_dir=debug_dir
    )
    
    print(f"\n✓ Edge refinement complete!")
    print(f"  - Median width: {sobel_result['median_width_cm']:.3f} cm")
    print(f"  - Success rate: {sobel_result['edge_detection_success_rate']:.1%}")
    print(f"  - Edge quality: {sobel_result['edge_quality']['overall_score']:.3f}")
    print(f"  - Samples: {sobel_result['num_samples']}")
    
    # Check debug images
    print("\n11. Checking debug output...")
    expected_images = [
        "01_landmark_axis.png",
        "02_ring_zone_roi.png",
        "03_roi_extraction.png",
        "04_sobel_left_to_right.png",
        "05_sobel_right_to_left.png",
        "06_gradient_magnitude.png",
        "07_edge_candidates.png",
        "08_selected_edges.png",
        "09_subpixel_refinement.png",
        "10_width_measurements.png",
        "11_width_distribution.png",  # May be missing if matplotlib unavailable
        "12_outlier_detection.png",
    ]
    
    found = 0
    for img_name in expected_images:
        img_path = os.path.join(debug_dir, img_name)
        if os.path.exists(img_path):
            size = os.path.getsize(img_path)
            print(f"  ✓ {img_name} ({size // 1024} KB)")
            found += 1
        else:
            if img_name == "11_width_distribution.png":
                print(f"  ⊘ {img_name} (matplotlib not available)")
            else:
                print(f"  ✗ {img_name} (missing)")
    
    print(f"\n✓ Found {found}/{len(expected_images)} debug images")
    
    # Test contour method for comparison
    print("\n12. Running contour method for comparison...")
    contour_result = compute_cross_section_width(
        finger_contour,
        axis_data,
        zone_data,
        px_per_cm
    )
    print(f"✓ Contour measurement: {contour_result['median_width_px'] / px_per_cm:.3f} cm")
    
    # Comparison
    print("\n" + "=" * 80)
    print("MEASUREMENT COMPARISON")
    print("=" * 80)
    contour_width_cm = contour_result['median_width_px'] / px_per_cm
    sobel_width_cm = sobel_result['median_width_cm']
    diff_cm = sobel_width_cm - contour_width_cm
    diff_pct = (diff_cm / contour_width_cm) * 100
    
    print(f"Contour (v0):  {contour_width_cm:.3f} cm")
    print(f"Sobel (v1):    {sobel_width_cm:.3f} cm")
    print(f"Difference:    {diff_cm:+.3f} cm ({diff_pct:+.1f}%)")
    
    print("\n" + "=" * 80)
    print("✓ PHASE 5 TEST COMPLETE")
    print("=" * 80)
    print(f"\nDebug images saved to: {debug_dir}/")
    print("Open these images to verify the debug visualization pipeline.")
    
    return True


if __name__ == "__main__":
    # Default test image
    test_image = "input/test_sample2.jpg"
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    success = test_edge_refinement_debug(test_image)
    sys.exit(0 if success else 1)
