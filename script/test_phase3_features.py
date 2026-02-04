#!/usr/bin/env python3
"""
Test Phase 3 features: Sub-pixel refinement and edge quality scoring.

Compares measurements with and without sub-pixel refinement.
"""

import cv2
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.card_detection import detect_credit_card, compute_scale_factor
from src.finger_segmentation import segment_hand, isolate_finger, clean_mask, get_finger_contour
from src.geometry import estimate_finger_axis, localize_ring_zone, compute_cross_section_width
from src.edge_refinement import refine_edges_sobel, should_use_sobel_measurement
from src.confidence import (
    compute_card_confidence,
    compute_finger_confidence,
    compute_measurement_confidence,
    compute_edge_quality_confidence,
    compute_overall_confidence
)


def test_phase3_features(image_path: str):
    """Test sub-pixel refinement and quality scoring."""
    print(f"\n{'='*70}")
    print(f"Phase 3 Features Test: Sub-Pixel Refinement & Quality Scoring")
    print(f"{'='*70}\n")

    # Load and process image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image")
        return

    h, w = image.shape[:2]

    # Get scale
    card_result = detect_credit_card(image)
    if card_result is None:
        print("Error: Card not detected")
        return

    px_per_cm, _ = compute_scale_factor(card_result["corners"])
    print(f"Scale: {px_per_cm:.2f} px/cm\n")

    # Get finger
    hand_data = segment_hand(image)
    if hand_data is None:
        print("Error: No hand detected")
        return

    finger_data = isolate_finger(hand_data, finger="auto", image_shape=(h, w), image=image)
    if finger_data is None:
        print("Error: Could not isolate finger")
        return

    mask = clean_mask(finger_data["mask"])
    contour = get_finger_contour(mask)
    landmarks = finger_data["landmarks"]

    print(f"Finger: {finger_data['finger_name']}\n")

    # Get axis and zone
    axis_data = estimate_finger_axis(mask, landmarks, method="auto")
    zone_data = localize_ring_zone(axis_data)

    # Test 1: Contour baseline
    print("="*70)
    print("Baseline: Contour Method (v0)")
    print("="*70)

    contour_result = compute_cross_section_width(contour, axis_data, zone_data, num_samples=20)
    contour_width_cm = contour_result["median_width_px"] / px_per_cm

    print(f"Width: {contour_width_cm:.4f} cm ({contour_result['median_width_px']:.2f} px)")
    print(f"Std dev: {contour_result['std_width_px']:.4f} px")
    print(f"Samples: {contour_result['num_samples']}\n")

    # Test 2: Sobel with sub-pixel refinement
    print("="*70)
    print("Phase 3: Sobel + Sub-Pixel Refinement")
    print("="*70)

    sobel_result = refine_edges_sobel(
        image=image,
        axis_data=axis_data,
        zone_data=zone_data,
        scale_px_per_cm=px_per_cm,
        finger_mask=mask,
        sobel_threshold=15.0,
        kernel_size=3,
    )

    sobel_width_cm = sobel_result["median_width_cm"]

    print(f"Width: {sobel_width_cm:.4f} cm ({sobel_result['median_width_px']:.2f} px)")
    print(f"Std dev: {sobel_result['std_width_px']:.4f} px")
    print(f"Samples: {sobel_result['num_samples']}")
    print(f"Sub-pixel: {sobel_result['subpixel_refinement_used']}")
    print(f"Success rate: {sobel_result['edge_detection_success_rate']*100:.1f}%\n")

    # Test 3: Edge Quality Scoring
    print("="*70)
    print("Edge Quality Assessment")
    print("="*70)

    eq = sobel_result["edge_quality"]

    print(f"Overall Score: {eq['overall_score']:.3f}")
    print(f"\nComponent Scores:")
    print(f"  Gradient Strength: {eq['gradient_strength_score']:.3f} (weight: 40%)")
    print(f"  Consistency:       {eq['consistency_score']:.3f} (weight: 30%)")
    print(f"  Smoothness:        {eq['smoothness_score']:.3f} (weight: 20%)")
    print(f"  Symmetry:          {eq['symmetry_score']:.3f} (weight: 10%)")

    print(f"\nRaw Metrics:")
    for key, value in eq['metrics'].items():
        print(f"  {key}: {value:.2f}")

    # Test 4: Auto Fallback Decision
    print(f"\n{'='*70}")
    print("Auto Fallback Decision")
    print("="*70)

    should_use, reason = should_use_sobel_measurement(sobel_result, contour_result)

    print(f"Use Sobel: {should_use}")
    print(f"Reason: {reason}")

    if should_use:
        print(f"✓ Sobel measurement quality acceptable, using Sobel result")
    else:
        print(f"✗ Sobel measurement quality insufficient, would fallback to contour")

    # Test 5: Comparison
    print(f"\n{'='*70}")
    print("Measurement Comparison")
    print("="*70)

    diff_cm = sobel_width_cm - contour_width_cm
    diff_pct = (diff_cm / contour_width_cm) * 100
    diff_px = sobel_result["median_width_px"] - contour_result["median_width_px"]

    print(f"Contour:    {contour_width_cm:.4f} cm ({contour_result['median_width_px']:.2f} px)")
    print(f"Sobel:      {sobel_width_cm:.4f} cm ({sobel_result['median_width_px']:.2f} px)")
    print(f"Difference: {diff_cm:+.4f} cm ({diff_pct:+.2f}%) ({diff_px:+.2f} px)")

    # Precision comparison
    print(f"\nPrecision (std dev):")
    print(f"Contour: {contour_result['std_width_px']:.4f} px")
    print(f"Sobel:   {sobel_result['std_width_px']:.4f} px")
    print(f"Ratio:   {sobel_result['std_width_px']/contour_result['std_width_px']:.2f}x")

    # Sub-pixel improvement estimation
    if sobel_result['subpixel_refinement_used']:
        print(f"\n✓ Sub-pixel refinement active")
        print(f"  Expected precision: <0.5 px (~{0.5/px_per_cm:.3f} cm)")
    else:
        print(f"\n✗ Sub-pixel refinement failed or disabled")

    # Test 6: Confidence Calculation Comparison
    print(f"\n{'='*70}")
    print("Confidence Calculation (v0 vs v1)")
    print("="*70)

    # Calculate component confidences
    card_conf = compute_card_confidence(card_result, _)
    finger_conf = compute_finger_confidence(hand_data, finger_data, np.sum(mask > 0), h * w)
    contour_meas_conf = compute_measurement_confidence(contour_result, contour_width_cm)
    sobel_meas_conf = compute_measurement_confidence(sobel_result, sobel_width_cm)
    edge_quality_conf = compute_edge_quality_confidence(sobel_result.get("edge_quality"))

    # v0 confidence (contour)
    conf_v0 = compute_overall_confidence(
        card_conf, finger_conf, contour_meas_conf,
        edge_method="contour"
    )

    # v1 confidence (sobel)
    conf_v1 = compute_overall_confidence(
        card_conf, finger_conf, sobel_meas_conf,
        edge_method="sobel",
        edge_quality_confidence=edge_quality_conf
    )

    print(f"\nComponent Confidences:")
    print(f"  Card:        {card_conf:.3f}")
    print(f"  Finger:      {finger_conf:.3f}")
    print(f"  Measurement: {contour_meas_conf:.3f} (contour) / {sobel_meas_conf:.3f} (sobel)")
    print(f"  Edge Quality: {edge_quality_conf:.3f} (sobel only)")

    print(f"\nOverall Confidence:")
    print(f"  v0 (Contour): {conf_v0['overall']:.3f} ({conf_v0['level']})")
    print(f"    Weights: Card 30%, Finger 30%, Measurement 40%")
    print(f"  v1 (Sobel):   {conf_v1['overall']:.3f} ({conf_v1['level']})")
    print(f"    Weights: Card 25%, Finger 25%, Edge 20%, Measurement 30%")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "input/test_sample2.jpg"

    test_phase3_features(image_path)
