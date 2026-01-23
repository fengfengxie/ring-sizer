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

from utils.image_quality import assess_image_quality
from utils.card_detection import detect_credit_card, compute_scale_factor
from utils.finger_segmentation import segment_hand, isolate_finger, clean_mask, get_finger_contour
from utils.geometry import estimate_finger_axis, localize_ring_zone, compute_cross_section_width
from utils.confidence import (
    compute_card_confidence,
    compute_finger_confidence,
    compute_measurement_confidence,
    compute_overall_confidence,
)
from utils.visualization import create_debug_visualization

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

    Returns:
        Output dictionary matching PRD specification
    """
    return {
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
) -> Dict[str, Any]:
    """
    Main measurement pipeline.

    Args:
        image: Input BGR image
        finger_index: Which finger to measure
        confidence_threshold: Minimum confidence threshold
        save_intermediate: Whether to save intermediate artifacts
        debug_path: Path to save debug visualization

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

    # Phase 3: Credit card detection & scale calibration
    card_result = detect_credit_card(image)

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

    # Phase 4: Hand & finger segmentation
    hand_data = segment_hand(image)

    if hand_data is None:
        print("No hand detected in image")
        return create_output(
            card_detected=True,
            finger_detected=False,
            scale_px_per_cm=px_per_cm,
            view_angle_ok=view_angle_ok,
            fail_reason="hand_not_detected",
        )

    print(f"Hand detected: {hand_data['handedness']}, confidence={hand_data['confidence']:.2f}")

    # Isolate the target finger
    h, w = image.shape[:2]
    finger_data = isolate_finger(hand_data, finger=finger_index, image_shape=(h, w))

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
    contour = get_finger_contour(cleaned_mask)

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
    try:
        width_data = compute_cross_section_width(
            contour=contour,
            axis_data=axis_data,
            zone_data=zone_data,
            num_samples=20,
        )

        # Convert to centimeters
        median_width_cm = width_data["median_width_px"] / px_per_cm
        mean_width_cm = width_data["mean_width_px"] / px_per_cm
        std_width_cm = width_data["std_width_px"] / px_per_cm

        print(f"Width measured: {width_data['num_samples']} samples, "
              f"median={median_width_cm:.2f}cm, std={std_width_cm:.3f}cm")

        # Sanity check: finger width should be in realistic range (1.4-2.4 cm)
        if median_width_cm < 1.0 or median_width_cm > 3.0:
            print(f"Warning: Measured width {median_width_cm:.2f}cm is outside realistic range")

    except Exception as e:
        print(f"Failed to measure finger width: {e}")
        return create_output(
            card_detected=True,
            finger_detected=True,
            scale_px_per_cm=px_per_cm,
            view_angle_ok=view_angle_ok,
            fail_reason="width_measurement_failed",
        )

    # Phase 8: Comprehensive confidence scoring
    # Calculate component confidences
    card_conf = compute_card_confidence(card_result, scale_confidence)

    # Calculate mask area for finger confidence
    mask_area = np.sum(cleaned_mask > 0)
    image_area = image.shape[0] * image.shape[1]
    finger_conf = compute_finger_confidence(hand_data, finger_data, mask_area, image_area)

    # Calculate measurement confidence
    measurement_conf = compute_measurement_confidence(width_data, median_width_cm)

    # Compute overall confidence
    confidence_breakdown = compute_overall_confidence(
        card_conf, finger_conf, measurement_conf
    )

    print(f"Confidence: {confidence_breakdown['overall']:.3f} ({confidence_breakdown['level']}) "
          f"[card={confidence_breakdown['card']:.2f}, "
          f"finger={confidence_breakdown['finger']:.2f}, "
          f"measurement={confidence_breakdown['measurement']:.2f}]")

    # Phase 9: Debug visualization
    if debug_path is not None:
        print(f"Generating debug visualization...")
        debug_image = create_debug_visualization(
            image=image,
            card_result=card_result,
            contour=contour,
            axis_data=axis_data,
            zone_data=zone_data,
            width_data=width_data,
            measurement_cm=median_width_cm,
            confidence=confidence_breakdown['overall'],
            scale_px_per_cm=px_per_cm,
        )

        # Save debug image
        Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(debug_path, debug_image)
        print(f"Debug visualization saved to: {debug_path}")

    return create_output(
        finger_diameter_cm=median_width_cm,
        confidence=confidence_breakdown['overall'],
        card_detected=True,
        finger_detected=True,
        scale_px_per_cm=px_per_cm,
        view_angle_ok=view_angle_ok,
        fail_reason=None,
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
