# Progress Log (v0) - Compressed

## Current Snapshot
- v0 pipeline is implemented and usable end-to-end from CLI.
- Core output contract is stable: measurement JSON + optional visualization.
- Main architecture modules are in place under `src/`.

## Major Milestones

### 2026-01-22 to 2026-01-23 - Core v0 delivery
- Implemented Phases 1-9:
  - CLI and project structure
  - image quality checks
  - credit-card detection + scale calibration
  - hand/finger segmentation
  - contour extraction + PCA axis estimation
  - ring-zone localization (15%-25%)
  - cross-section width measurement (median)
  - confidence scoring
  - debug overlay rendering
- Outcome: full baseline measurement flow operational.

### 2026-02-02 - Card detection visibility + quality tweaks
- Added rich card-detection debug pipeline images.
- Reduced debug image size significantly via downsampling/compression.
- Lowered blur threshold (50 -> 20) to avoid rejecting valid phone images.
- Simplified corner refinement path (sub-pixel refinement retained).

### 2026-02-03 - Refactor and documentation pass
- Standardized directories:
  - `docs -> doc`, `samples -> input`, `outputs -> output`, `utils -> src`, etc.
- Centralized visualization constants into `src/viz_constants.py`.
- Improved debug readability (font/layout updates).
- Migrated algorithm documentation to modular structure:
  - `doc/v0/algorithms/README.md`
  - per-phase algorithm docs.

### 2026-02-03 - Finger segmentation accuracy upgrade
- Replaced polygon-only isolation as primary path with pixel-level isolation (ROI ∩ hand mask + component selection).
- Kept polygon method as fallback.
- Added detailed stage debug outputs for segmentation.
- Added documentation for segmentation behavior and debug interpretation.

### 2026-02-03 - Debug clarity bugfix
- Clarified/expanded pixel-level debug stages to show component selection explicitly.
- Reduced confusion around visually similar intermediate masks.

## v0 Technical State
- Measurement method: contour intersections over sampled perpendicular cross-sections.
- Ring zone: palm-side 15%-25% (baseline mode).
- Confidence model: card/finger/measurement weighted scoring.
- Failure reasons are surfaced through structured output.

## Notes for Continuation
- v0 serves as fallback/reference baseline for v1 edge-refinement modes.
- v0 behavior is intentionally retained for compatibility and robustness.

