# 04 - Hand and Finger Segmentation

Module: `src/finger_segmentation.py`

## Purpose
Detect hand landmarks, normalize orientation, isolate the target finger, and produce a clean mask/landmarks for geometry + measurement.

## Current behavior
- MediaPipe hand landmarks detection.
- Orientation normalization to canonical pose (wrist down, fingers up).
- Finger selection by `--finger-index` (`index` default, `auto` supported).
- Finger isolation pipeline with cleanup and contour extraction support.

## Outputs used downstream
- Canonical image (for robust Sobel orientation assumptions).
- Target-finger landmarks.
- Finger mask (used in contour path and geometry steps).

## Failure points
- `hand_not_detected`
- `finger_isolation_failed`
- `finger_mask_too_small`
- `contour_extraction_failed`

## Debug
With `--debug`, intermediate segmentation stages are written under `output/*finger_segmentation_debug*/`.
