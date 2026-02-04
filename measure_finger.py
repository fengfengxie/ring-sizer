#!/usr/bin/env python3
"""
Finger Outer Diameter Measurement Tool

Measures the outer width (diameter) of a finger at the ring-wearing zone
using a single RGB image with a credit card as a physical size reference.

Usage:
    python measure_finger.py --input image.jpg --output result.json [--debug debug.png]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Literal

import cv2
import numpy as np

from src.image_quality import assess_image_quality
from src.card_detection import detect_credit_card, compute_scale_factor
from src.finger_segmentation import segment_hand, isolate_finger, clean_mask, get_finger_contour
from src.geometry import estimate_finger_axis, localize_ring_zone, compute_cross_section_width
from src.edge_refinement import refine_edges_sobel, should_use_sobel_measurement, compare_edge_methods
from src.confidence import (
    compute_card_confidence,
    compute_finger_confidence,
    compute_measurement_confidence,
    compute_edge_quality_confidence,
    compute_overall_confidence,
)
from src.visualization import create_debug_visualization

# Type alias for finger selection
FingerIndex = Literal["auto", "index", "middle", "ring", "pinky"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Measure finger outer diameter from an image with credit card reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python measure_finger.py --input photo.jpg --output result.json
    python measure_finger.py --input photo.jpg --output result.json --debug overlay.png
    python measure_finger.py --input photo.jpg --output result.json --finger-index ring
        """,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image (JPG/PNG)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file",
    )

    # Optional arguments
    parser.add_argument(
        "--debug",
        type=str,
        default=None,
        help="Path to save debug visualization (PNG)",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate processing artifacts",
    )
    parser.add_argument(
        "--finger-index",
        type=str,
        choices=["auto", "index", "middle", "ring", "pinky"],
        default="auto",
        help="Which finger to measure (default: auto-detect)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence threshold (default: 0.7)",
    )

    # v1 edge refinement options
    parser.add_argument(
        "--edge-method",
        type=str,
        default="auto",
        choices=["auto", "contour", "sobel", "compare"],
        help="Edge detection method: auto (quality-based), contour (v0), sobel (v1), compare (both) (default: auto)",
    )
    parser.add_argument(
        "--sobel-threshold",
        type=float,
        default=15.0,
        help="Minimum gradient magnitude for valid edge (default: 15.0)",
    )
    parser.add_argument(
        "--sobel-kernel-size",
        type=int,
        default=3,
        choices=[3, 5, 7],
        help="Sobel kernel size (default: 3)",
    )
    parser.add_argument(
        "--no-subpixel",
        action="store_true",
        help="Disable sub-pixel edge refinement",
    )

    # Testing/debugging options
    parser.add_argument(
        "--skip-card-detection",
        action="store_true",
        help="[TESTING ONLY] Skip card detection and use dummy scale (allows testing finger segmentation without card)",
    )

    return parser.parse_args()


def validate_input(input_path: str) -> Optional[str]:
    """
    Validate input file exists and is a supported image format.

    Args:
        input_path: Path to input image

    Returns:
        Error message if validation fails, None if valid
    """
    path = Path(input_path)

    if not path.exists():
        return f"Input file not found: {input_path}"

    if not path.is_file():
        return f"Input path is not a file: {input_path}"

    suffix = path.suffix.lower()
    if suffix not in [".jpg", ".jpeg", ".png"]:
        return f"Unsupported image format: {suffix}. Use JPG or PNG."

    return None


def load_image(input_path: str) -> Optional[np.ndarray]:
    """
    Load image from file.

    Args:
        input_path: Path to input image

    Returns:
        BGR image as numpy array, or None if load fails
    """
    image = cv2.imread(input_path)
    return image


def create_output(
    finger_diameter_cm: Optional[float] = None,
    confidence: float = 0.0,
    scale_px_per_cm: Optional[float] = None,
    card_detected: bool = False,
    finger_detected: bool = False,
    view_angle_ok: bool = True,
    fail_reason: Optional[str] = None,
    edge_method_used: Optional[str] = None,
    method_comparison: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create output dictionary in the specified format.

    Args:
        finger_diameter_cm: Measured finger diameter in cm
        confidence: Confidence score [0, 1]
        scale_px_per_cm: Computed scale factor
        card_detected: Whether credit card was detected
        finger_detected: Whether finger was detected
        view_angle_ok: Whether view angle is acceptable
        fail_reason: Reason for failure if applicable
        edge_method_used: Edge detection method used (v1)
        method_comparison: Comparison data when using compare mode (v1)

    Returns:
        Output dictionary matching PRD specification
    """
    output = {
        "finger_outer_diameter_cm": float(finger_diameter_cm) if finger_diameter_cm is not None else None,
        "confidence": float(round(confidence, 3)),
        "scale_px_per_cm": round(float(scale_px_per_cm), 2) if scale_px_per_cm is not None else None,
        "quality_flags": {
            "card_detected": bool(card_detected),
            "finger_detected": bool(finger_detected),
            "view_angle_ok": bool(view_angle_ok),
        },
        "fail_reason": fail_reason,
    }

    # Add v1 fields if applicable
    if edge_method_used is not None:
        output["edge_method_used"] = edge_method_used

    if method_comparison is not None:
        output["method_comparison"] = method_comparison

    return output


def save_output(output: Dict[str, Any], output_path: str) -> None:
    """Save output dictionary to JSON file."""
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def measure_finger(
    image: np.ndarray,
    finger_index: FingerIndex = "auto",
    confidence_threshold: float = 0.7,
    save_intermediate: bool = False,
    debug_path: Optional[str] = None,
    edge_method: str = "auto",
    sobel_threshold: float = 15.0,
    sobel_kernel_size: int = 3,
    use_subpixel: bool = True,
    skip_card_detection: bool = False,
) -> Dict[str, Any]:
    """
    Main measurement pipeline.

    Args:
        image: Input BGR image
        finger_index: Which finger to measure
        confidence_threshold: Minimum confidence threshold
        save_intermediate: Whether to save intermediate artifacts
        debug_path: Path to save debug visualization
        edge_method: Edge detection method (auto, contour, sobel, compare)
        sobel_threshold: Minimum gradient magnitude for valid edge
        sobel_kernel_size: Sobel kernel size (3, 5, or 7)
        use_subpixel: Enable sub-pixel edge refinement

    Returns:
        Output dictionary with measurement results
    """
    # Phase 2: Image quality check
    quality = assess_image_quality(image)
    print(f"Image quality: blur={quality['blur_score']:.1f}, "
          f"brightness={quality['brightness']:.1f}, "
          f"contrast={quality['contrast']:.1f}")

    if not quality["passed"]:
        for issue in quality["issues"]:
            print(f"  Warning: {issue}")
        return create_output(fail_reason=quality["fail_reason"])

    # Phase 3: Hand & finger segmentation (MOVED BEFORE CARD DETECTION)
    # This allows us to rotate the image to canonical orientation first
    # Create finger segmentation debug subdirectory if debug enabled
    finger_debug_dir = None
    if debug_path is not None:
        from pathlib import Path
        finger_debug_dir = str(Path(debug_path).parent / "finger_segmentation_debug")

    hand_data = segment_hand(image, debug_dir=finger_debug_dir)

    if hand_data is None:
        print("No hand detected in image")
        return create_output(
            card_detected=False,  # Card not yet detected
            finger_detected=False,
            fail_reason="hand_not_detected",
        )

    print(f"Hand detected: {hand_data['handedness']}, confidence={hand_data['confidence']:.2f}")
    if "orientation_rotation" in hand_data:
        print(f"Hand orientation normalized: {hand_data['orientation_rotation']}° rotation applied")
    
    # Use canonical image for all downstream processing
    # This ensures finger edges are vertical for optimal Sobel detection
    if "canonical_image" in hand_data:
        image_canonical = hand_data["canonical_image"]
        print(f"Using canonical orientation image: {image_canonical.shape[1]}x{image_canonical.shape[0]}")
    else:
        image_canonical = image  # Fallback if not available
        print("Warning: Canonical image not available, using original")

    # Phase 4: Credit card detection & scale calibration (NOW ON CANONICAL IMAGE)
    # Create card detection debug subdirectory if debug enabled
    card_debug_dir = None
    if debug_path is not None:
        from pathlib import Path
        card_debug_dir = str(Path(debug_path).parent / "card_detection_debug")

    # Allow skipping card detection for testing finger segmentation
    if skip_card_detection:
        print("⚠️  TESTING MODE: Skipping card detection (using dummy scale factor)")
        card_result = None
        px_per_cm = 100.0  # Dummy scale: 100 pixels/cm (measurements will be inaccurate)
        scale_confidence = 0.5  # Low confidence to indicate dummy value
        view_angle_ok = True
        card_detected = False
    else:
        card_result = detect_credit_card(image_canonical, debug_dir=card_debug_dir)

        if card_result is None:
            print("Credit card not detected in image")
            return create_output(
                card_detected=False,
                fail_reason="card_not_detected",
            )

        # Compute scale factor
        px_per_cm, scale_confidence = compute_scale_factor(card_result["corners"])
        print(f"Card detected: {card_result['width_px']:.0f}x{card_result['height_px']:.0f}px, "
              f"aspect={card_result['aspect_ratio']:.3f}, confidence={card_result['confidence']:.2f}")
        print(f"Scale: {px_per_cm:.2f} px/cm (confidence={scale_confidence:.2f})")

        # Check for excessive perspective distortion (view angle)
        view_angle_ok = scale_confidence > 0.9
        card_detected = True

    # Phase 5: Finger isolation (hand already segmented in Phase 3)
    h_can, w_can = image_canonical.shape[:2]
    finger_data = isolate_finger(hand_data, finger=finger_index, image_shape=(h_can, w_can),
                                 image=image_canonical, debug_dir=finger_debug_dir)

    if finger_data is None:
        print(f"Could not isolate finger: {finger_index}")
        return create_output(
            card_detected=True,
            finger_detected=False,
            scale_px_per_cm=px_per_cm,
            view_angle_ok=view_angle_ok,
            fail_reason="finger_isolation_failed",
        )

    print(f"Finger isolated: {finger_data['finger_name']}")

    # Clean the finger mask
    cleaned_mask = clean_mask(finger_data["mask"])

    if cleaned_mask is None:
        print("Finger mask too small or invalid")
        return create_output(
            card_detected=True,
            finger_detected=False,
            scale_px_per_cm=px_per_cm,
            view_angle_ok=view_angle_ok,
            fail_reason="finger_mask_too_small",
        )

    # Extract finger contour
    contour = get_finger_contour(cleaned_mask, debug_dir=finger_debug_dir)

    if contour is None:
        print("Could not extract finger contour")
        return create_output(
            card_detected=True,
            finger_detected=False,
            scale_px_per_cm=px_per_cm,
            view_angle_ok=view_angle_ok,
            fail_reason="contour_extraction_failed",
        )

    print(f"Finger contour extracted: {len(contour)} points")

    # Phase 5: Estimate finger axis using PCA
    try:
        axis_data = estimate_finger_axis(
            mask=cleaned_mask,
            landmarks=finger_data.get("landmarks"),
        )
        print(f"Finger axis estimated: length={axis_data['length']:.1f}px, "
              f"center=({axis_data['center'][0]:.0f}, {axis_data['center'][1]:.0f})")
    except Exception as e:
        print(f"Failed to estimate finger axis: {e}")
        return create_output(
            card_detected=True,
            finger_detected=True,
            scale_px_per_cm=px_per_cm,
            view_angle_ok=view_angle_ok,
            fail_reason="axis_estimation_failed",
        )

    # Phase 6: Localize ring-wearing zone
    try:
        zone_data = localize_ring_zone(axis_data)
        zone_length_cm = zone_data["length"] / px_per_cm
        print(f"Ring zone localized: {zone_data['start_pct']*100:.0f}%-{zone_data['end_pct']*100:.0f}% "
              f"from palm, length={zone_data['length']:.1f}px ({zone_length_cm:.2f}cm)")
    except Exception as e:
        print(f"Failed to localize ring zone: {e}")
        return create_output(
            card_detected=True,
            finger_detected=True,
            scale_px_per_cm=px_per_cm,
            view_angle_ok=view_angle_ok,
            fail_reason="zone_localization_failed",
        )

    # Phase 7: Measure finger width at ring zone

    # Phase 7a: Contour-based measurement (v0 method)
    try:
        contour_measurement = compute_cross_section_width(
            contour=contour,
            axis_data=axis_data,
            zone_data=zone_data,
            num_samples=20,
        )

        contour_width_cm = contour_measurement["median_width_px"] / px_per_cm
        print(f"Contour width: {contour_width_cm:.4f}cm "
              f"({contour_measurement['num_samples']} samples, "
              f"std={contour_measurement['std_width_px']:.2f}px)")

    except Exception as e:
        print(f"Failed to measure finger width (contour): {e}")
        return create_output(
            card_detected=True,
            finger_detected=True,
            scale_px_per_cm=px_per_cm,
            view_angle_ok=view_angle_ok,
            fail_reason="width_measurement_failed",
            edge_method_used="contour",
        )

    # Phase 7b: Sobel-based measurement (v1 method)
    sobel_measurement = None
    sobel_failed = False

    if edge_method in ["sobel", "auto", "compare"]:
        try:
            print(f"Running Sobel edge refinement (threshold={sobel_threshold}, kernel={sobel_kernel_size})...")
            
            # Create debug directory for edge refinement if main debug is enabled
            edge_debug_dir = None
            if debug_path is not None:
                edge_debug_dir = str(Path(debug_path).parent / "edge_refinement_debug")
            
            sobel_measurement = refine_edges_sobel(
                image=image_canonical,  # Use canonical orientation
                axis_data=axis_data,
                zone_data=zone_data,
                scale_px_per_cm=px_per_cm,
                finger_mask=cleaned_mask,
                finger_landmarks=finger_data.get("landmarks"),
                sobel_threshold=sobel_threshold,
                kernel_size=sobel_kernel_size,
                debug_dir=edge_debug_dir,
            )

            sobel_width_cm = sobel_measurement["median_width_cm"]
            print(f"Sobel width: {sobel_width_cm:.4f}cm "
                  f"({sobel_measurement['num_samples']} samples, "
                  f"std={sobel_measurement['std_width_px']:.2f}px, "
                  f"quality={sobel_measurement['edge_quality']['overall_score']:.3f})")

        except Exception as e:
            print(f"Sobel edge refinement failed: {e}")
            sobel_failed = True
            if edge_method == "sobel":
                # User explicitly requested Sobel, fail if it doesn't work
                return create_output(
                    card_detected=True,
                    finger_detected=True,
                    scale_px_per_cm=px_per_cm,
                    view_angle_ok=view_angle_ok,
                    fail_reason="sobel_edge_refinement_failed",
                    edge_method_used="sobel",
                )

    # Select measurement method based on edge_method flag
    method_comparison_data = None

    if edge_method == "contour":
        # Use contour method only
        final_measurement = contour_measurement
        median_width_cm = contour_width_cm
        edge_method_used = "contour"

    elif edge_method == "sobel":
        # Use Sobel method only (already handled failure case above)
        final_measurement = sobel_measurement
        median_width_cm = sobel_measurement["median_width_cm"]
        edge_method_used = "sobel"

    elif edge_method == "auto":
        # Automatic selection based on quality
        if sobel_measurement and not sobel_failed:
            should_use_sobel, reason = should_use_sobel_measurement(sobel_measurement, contour_measurement)

            if should_use_sobel:
                final_measurement = sobel_measurement
                median_width_cm = sobel_measurement["median_width_cm"]
                edge_method_used = "sobel"
                print(f"Auto-selected: Sobel (reason: {reason})")
            else:
                final_measurement = contour_measurement
                median_width_cm = contour_width_cm
                edge_method_used = "contour_fallback"
                print(f"Auto-selected: Contour fallback (reason: {reason})")
        else:
            # Sobel failed, use contour
            final_measurement = contour_measurement
            median_width_cm = contour_width_cm
            edge_method_used = "contour_fallback"
            print(f"Auto-selected: Contour (Sobel not available)")

    elif edge_method == "compare":
        # Comparison mode: prefer Sobel if available, include comparison data
        if sobel_measurement and not sobel_failed:
            method_comparison_data = compare_edge_methods(
                contour_measurement, sobel_measurement, px_per_cm
            )

            # Prefer Sobel in compare mode for output
            final_measurement = sobel_measurement
            median_width_cm = sobel_measurement["median_width_cm"]
            edge_method_used = "compare"

            print(f"Method comparison:")
            print(f"  Contour: {method_comparison_data['contour']['width_cm']:.4f}cm")
            print(f"  Sobel:   {method_comparison_data['sobel']['width_cm']:.4f}cm")
            print(f"  Diff:    {method_comparison_data['difference']['relative_pct']:+.2f}%")
            print(f"  Recommendation: {method_comparison_data['recommendation']['preferred_method']}")
        else:
            # Sobel failed, can't compare
            final_measurement = contour_measurement
            median_width_cm = contour_width_cm
            edge_method_used = "contour"
            print(f"Compare mode: Sobel failed, using contour only")

    # Sanity check: finger width should be in realistic range (1.4-2.4 cm)
    if median_width_cm < 1.0 or median_width_cm > 3.0:
        print(f"Warning: Measured width {median_width_cm:.2f}cm is outside realistic range")

    # Phase 8: Comprehensive confidence scoring
    # Calculate component confidences
    if card_result is not None:
        card_conf = compute_card_confidence(card_result, scale_confidence)
    else:
        # Dummy card confidence when card detection skipped (testing mode)
        card_conf = scale_confidence  # Use dummy scale confidence (0.5)

    # Calculate mask area for finger confidence
    mask_area = np.sum(cleaned_mask > 0)
    image_area = image.shape[0] * image.shape[1]
    finger_conf = compute_finger_confidence(hand_data, finger_data, mask_area, image_area)

    # Calculate measurement confidence
    measurement_conf = compute_measurement_confidence(final_measurement, median_width_cm)

    # Calculate edge quality confidence (v1)
    edge_quality_conf = None
    if edge_method_used in ["sobel", "compare"]:
        edge_quality_conf = compute_edge_quality_confidence(
            final_measurement.get("edge_quality")
        )

    # Compute overall confidence (v0 or v1 based on edge method)
    confidence_breakdown = compute_overall_confidence(
        card_conf,
        finger_conf,
        measurement_conf,
        edge_method=edge_method_used if edge_method_used in ["contour", "sobel"] else "contour",
        edge_quality_confidence=edge_quality_conf,
    )

    # Print confidence breakdown
    conf_parts = [
        f"card={confidence_breakdown['card']:.2f}",
        f"finger={confidence_breakdown['finger']:.2f}",
        f"measurement={confidence_breakdown['measurement']:.2f}",
    ]
    if confidence_breakdown.get('edge_quality') is not None:
        conf_parts.append(f"edge={confidence_breakdown['edge_quality']:.2f}")

    print(f"Confidence: {confidence_breakdown['overall']:.3f} ({confidence_breakdown['level']}) "
          f"[{', '.join(conf_parts)}]")

    # Phase 9: Debug visualization
    if debug_path is not None:
        print(f"Generating debug visualization...")
        debug_image = create_debug_visualization(
            image=image_canonical,  # Use canonical orientation
            card_result=card_result,
            contour=contour,
            axis_data=axis_data,
            zone_data=zone_data,
            width_data=final_measurement,
            measurement_cm=median_width_cm,
            confidence=confidence_breakdown['overall'],
            scale_px_per_cm=px_per_cm,
        )

        # Save debug image
        Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(debug_path, debug_image)
        print(f"Debug visualization saved to: {debug_path} (canonical orientation)")


    return create_output(
        finger_diameter_cm=median_width_cm,
        confidence=confidence_breakdown['overall'],
        card_detected=card_detected,
        finger_detected=True,
        scale_px_per_cm=px_per_cm,
        view_angle_ok=view_angle_ok,
        fail_reason=None,
        edge_method_used=edge_method_used,
        method_comparison=method_comparison_data,
    )


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Validate input
    error = validate_input(args.input)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    # Load image
    image = load_image(args.input)
    if image is None:
        print(f"Error: Failed to load image: {args.input}", file=sys.stderr)
        return 1

    print(f"Loaded image: {args.input} ({image.shape[1]}x{image.shape[0]})")

    # Run measurement pipeline
    result = measure_finger(
        image=image,
        finger_index=args.finger_index,
        confidence_threshold=args.confidence_threshold,
        save_intermediate=args.save_intermediate,
        debug_path=args.debug,
        edge_method=args.edge_method,
        sobel_threshold=args.sobel_threshold,
        sobel_kernel_size=args.sobel_kernel_size,
        use_subpixel=not args.no_subpixel,
        skip_card_detection=args.skip_card_detection,
    )

    # Save output
    save_output(result, args.output)
    print(f"Results saved to: {args.output}")

    # Report result
    if result["fail_reason"]:
        print(f"Measurement failed: {result['fail_reason']}")
        return 1
    else:
        print(f"Finger diameter: {result['finger_outer_diameter_cm']} cm")
        print(f"Confidence: {result['confidence']}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
