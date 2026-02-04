#!/usr/bin/env python3
"""
Test Phase 4 integration: Method comparison and CLI integration.

Tests all edge detection methods through the main pipeline.
"""

import cv2
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measure_finger import measure_finger


def test_all_edge_methods(image_path: str):
    """Test all edge detection methods."""
    print(f"\n{'='*70}")
    print(f"Phase 4 Integration Test: All Edge Methods")
    print(f"{'='*70}\n")

    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image")
        return

    methods = ["contour", "sobel", "auto", "compare"]

    results = {}

    for method in methods:
        print(f"\n{'='*70}")
        print(f"Testing edge method: {method}")
        print(f"{'='*70}\n")

        try:
            result = measure_finger(
                image=image,
                finger_index="auto",
                confidence_threshold=0.7,
                save_intermediate=False,
                debug_path=None,
                edge_method=method,
                sobel_threshold=15.0,
                sobel_kernel_size=3,
                use_subpixel=True,
            )

            results[method] = result

            # Print key results
            print(f"\nResult for {method}:")
            print(f"  Diameter: {result['finger_outer_diameter_cm']} cm")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Edge method used: {result.get('edge_method_used', 'N/A')}")

            if result.get('method_comparison'):
                comp = result['method_comparison']
                print(f"  Comparison:")
                print(f"    Contour: {comp['contour']['width_cm']:.4f}cm")
                print(f"    Sobel:   {comp['sobel']['width_cm']:.4f}cm")
                print(f"    Diff:    {comp['difference']['relative_pct']:+.2f}%")
                print(f"    Recommendation: {comp['recommendation']['preferred_method']}")

        except Exception as e:
            print(f"Error testing {method}: {e}")
            import traceback
            traceback.print_exc()

    # Summary comparison
    print(f"\n{'='*70}")
    print(f"Summary: All Methods")
    print(f"{'='*70}\n")

    print(f"{'Method':<15} {'Diameter (cm)':<15} {'Confidence':<12} {'Edge Used':<20}")
    print("-" * 70)

    for method in methods:
        if method in results:
            r = results[method]
            diameter = r.get('finger_outer_diameter_cm', 'N/A')
            confidence = r.get('confidence', 'N/A')
            edge_used = r.get('edge_method_used', 'N/A')

            print(f"{method:<15} {diameter if diameter == 'N/A' else f'{diameter:.4f}':<15} "
                  f"{confidence if confidence == 'N/A' else f'{confidence:.3f}':<12} {edge_used:<20}")

    print(f"\n{'='*70}\n")

    # Save results to JSON
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for method, result in results.items():
        output_path = f"{output_dir}/phase4_test_{method}.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved {method} result to: {output_path}")


def test_cli_flags():
    """Test that all CLI flags are accepted."""
    print(f"\n{'='*70}")
    print(f"Testing CLI Flag Parsing")
    print(f"{'='*70}\n")

    import argparse
    from measure_finger import parse_args

    # Save original sys.argv
    original_argv = sys.argv

    try:
        # Test basic flags
        sys.argv = [
            'measure_finger.py',
            '--input', 'test.jpg',
            '--output', 'test.json',
            '--edge-method', 'auto',
            '--sobel-threshold', '20.0',
            '--sobel-kernel-size', '5',
            '--no-subpixel',
        ]

        args = parse_args()
        print("✓ CLI flags parsed successfully")
        print(f"  edge_method: {args.edge_method}")
        print(f"  sobel_threshold: {args.sobel_threshold}")
        print(f"  sobel_kernel_size: {args.sobel_kernel_size}")
        print(f"  no_subpixel: {args.no_subpixel}")

    except Exception as e:
        print(f"✗ CLI flag parsing failed: {e}")

    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "input/test_sample2.jpg"

    # Test CLI flags
    test_cli_flags()

    # Test all edge methods
    test_all_edge_methods(image_path)
