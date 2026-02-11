# Progress Log (v1) - Compressed

## Current Snapshot
- v1 edge-refinement features are implemented and integrated.
- Default operation supports `--edge-method auto` with fallback to contour.
- Output and CLI remain backward compatible with v0.

## Delivery Summary by Phase

### Phase 0 (Planning) - 2026-02-04
- Completed v1 PRD and implementation plan.
- Defined goals:
  - landmark-based axis
  - Sobel edge refinement
  - sub-pixel localization
  - quality-based fallback
  - comparison mode

### Phase 1 (Landmark Axis) - 2026-02-04
- Added landmark-first axis estimation in `src/geometry.py` with PCA fallback.
- Added landmark quality validation.
- Added landmark-based ring-zone localization option.
- Outcome: robust axis selection with transparent fallback behavior.

### Phase 2 (Sobel Core) - 2026-02-04
- Added `src/edge_refinement.py` with:
  - ROI extraction
  - Sobel gradient generation
  - per-row edge detection
  - width computation and outlier filtering
- Integrated Sobel path into main pipeline.

### Phase 3 (Quality + Fallback) - 2026-02-04
- Added sub-pixel edge localization (parabolic refinement).
- Added edge-quality scoring (strength/consistency/smoothness/symmetry).
- Added `should_use_sobel_measurement()` fallback gating logic.
- Extended confidence model to support Sobel edge-quality component.

### Phase 4 (Integration + CLI) - 2026-02-04
- Added/finished CLI flags:
  - `--edge-method {auto,contour,sobel,compare}`
  - `--sobel-threshold`
  - `--sobel-kernel-size`
  - `--no-subpixel`
- Added compare mode output structure.
- Preserved backward compatibility for existing consumers.

### Phase 5 (Debug/Visualization) - 2026-02-04 onward
- Added staged edge-refinement debug outputs and comprehensive overlay support.
- Unified and improved visual diagnostics for edge selection and measurement.

### Phase 6 (Validation/Refinement) - ongoing
- Iterative tuning performed on available sample images.
- Fallback behavior and quality gating validated functionally.
- Full benchmark/ground-truth campaign remains the next structured validation step.

## Post-Phase Improvements

### 2026-02-04 - Orientation + Sobel correctness
- Added canonical hand-orientation normalization before downstream processing.
- Fixed Sobel filter-orientation bug after rotation normalization.
- Added axis-constrained edge logic and improved overlay diagnostics.

### 2026-02-04 - Refactor quality pass
- Centralized thresholds/constants:
  - `src/edge_refinement_constants.py`
  - `src/geometry_constants.py`
  - `src/confidence_constants.py`
- Replaced ad-hoc prints with logging in core modules.

### 2026-02-05 - Finger selection behavior
- Made orientation detection finger-aware and aligned with `--finger-index`.
- Default finger selection set to `index`; `auto` preserved.
- Updated docs/examples.

### 2026-02-05 to 2026-02-10 - Reliability + UX
- Switched Sobel ROI constraint to full-ROI mask (not segmentation mask clipping).
- Card detection corner extraction moved to `minAreaRect`.
- Result PNG generation made automatic alongside JSON.
- `--debug` converted to boolean (intermediate debug folders only).
- Updated `script/test.sh` to new debug behavior.

### 2026-02-11 - Consistency fixes
- Wired `--no-subpixel` through to Sobel measurement path.
- Fixed compare-mode confidence weighting to use Sobel weighting when appropriate.
- Prevented Sobel overlay from rendering when auto mode falls back to contour.
- Corrected `card_detected` consistency in skip-card failure paths.
- Removed unused Sobel `finger_mask` parameters and cleaned related docs/comments.

### 2026-02-11 - Directional half-ROI Sobel gating
- Added directional Sobel maps in `src/edge_refinement.py`:
  - `left_to_right` responses retained only on ROI right half
  - `right_to_left` responses retained only on ROI left half
- Updated edge selection to use direction-specific gradients per side:
  - left edge search uses right-to-left map
  - right edge search uses left-to-right map
- Updated stage debug outputs:
  - `04_sobel_left_to_right` now visualizes half-gated `left_to_right`
  - `05_sobel_right_to_left` now visualizes half-gated `right_to_left`
- Goal: reduce nearby non-target finger edge contamination in Sobel mode.

### 2026-02-11 - Edge debug cleanup (06a-06h removed)
- Removed generation of experimental filter-comparison debug stages:
  - `06a_filter_gaussian` through `06h_filter_unsharp`
- Removed unused helper implementation from `src/debug_observer.py`:
  - deleted `draw_gradient_filtering_techniques()`
- Simplified Sobel debug output to core stages only (`04`, `05`, `06`, `07`, `09+`).

### 2026-02-11 - Finger segmentation debug cleanup (remove 02a)
- Removed `02a_orientation_detection` debug stage from `src/finger_segmentation.py`.
- Deleted unused orientation-overlay drawing branch in `normalize_hand_orientation()`
  (arrow + text rendering), while keeping orientation normalization behavior unchanged.
- Removed now-unused visualization constants import tied only to that stage.

### 2026-02-11 - Web demo default sample quick start
- Copied `input/sample-02-05/10.jpg` into tracked path:
  - `web_demo/static/examples/default_sample.jpg`
- Web demo now shows this sample as the default input preview on load.
- Added one-click sample execution endpoint and UI flow:
  - backend: `POST /api/measure-default`
  - frontend button: `Run Sample Image`
- Goal: faster first run and clearer guidance for what kind of photos users should upload.

## Files Most Affected in v1
- `measure_finger.py`
- `src/edge_refinement.py`
- `src/geometry.py`
- `src/confidence.py`
- `src/debug_observer.py`
- `src/finger_segmentation.py`

## Known Operational Note
- In restricted/sandbox environments, MediaPipe may fail at runtime due to OpenGL/GPU service availability. This is environment-related and separate from algorithm logic.

## Next Recommended Step
- Run a formal validation pass across sample/ground-truth set:
  - compare `contour` vs `sobel` vs `auto`
  - summarize MAE, variance, and fallback rate
  - publish `doc/v1/validation-summary.md`.
