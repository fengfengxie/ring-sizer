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
        "finger_outer_diameter_cm": finger_diameter_cm,
        "confidence": round(confidence, 3),
        "scale_px_per_cm": round(scale_px_per_cm, 2) if scale_px_per_cm else None,
        "quality_flags": {
            "card_detected": card_detected,
            "finger_detected": finger_detected,
            "view_angle_ok": view_angle_ok,
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

    # TODO: Implement remaining pipeline in subsequent phases
    # Phase 3: Credit card detection & scale calibration
    # Phase 4: Hand & finger segmentation
    # Phase 5: Finger contour & axis estimation
    # Phase 6: Ring-wearing zone localization
    # Phase 7: Width measurement
    # Phase 8: Confidence scoring
    # Phase 9: Debug visualization

    # For now, return a placeholder indicating not implemented
    return create_output(
        fail_reason="Pipeline not yet implemented. Coming in Phase 3-9.",
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
