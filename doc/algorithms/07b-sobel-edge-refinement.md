# 07b - Sobel Edge Refinement

Module: `src/edge_refinement.py`

## Purpose
Provide a gradient-based width measurement path as an alternative to contour intersection.

## Current behavior
1. Extract ROI around ring zone (`extract_ring_zone_roi`).
   - Intentional sizing: `1.5x` zone length wide, `0.5x` zone length tall.
2. Compute Sobel gradients (`apply_sobel_filters`).
3. Detect left/right edges per row (`detect_edges_per_row`).
4. Optionally apply sub-pixel refinement (`refine_edge_subpixel`).
5. Aggregate widths with outlier filtering (`measure_width_from_edges`).
6. Score edge quality (`compute_edge_quality_score`).
7. Decide Sobel vs contour fallback in auto mode (`should_use_sobel_measurement`).

## Integration with CLI
- `--edge-method sobel`: require Sobel result.
- `--edge-method auto` (default): Sobel + quality checks, fallback to contour when needed.
- `--edge-method compare`: include both method summaries.
- `--sobel-threshold`, `--sobel-kernel-size`, `--no-subpixel` are active.

## Output fields
When Sobel path runs, output may include:
- `edge_method_used` (`sobel`, `contour_fallback`, or `compare` depending on mode)
- `method_comparison` (in compare mode)

## Debug
With `--debug`, intermediate Sobel stages are written under `output/*edge_refinement_debug*/`.
