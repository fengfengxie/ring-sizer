# 02 - Credit Card Detection

Module: `src/card_detection.py`

## Purpose
Detect a credit card in the image and provide corners for scale calibration.

## Current behavior
- Runs multi-strategy candidate extraction (edge/threshold/color routes).
- Selects best candidate using geometric filtering + scoring.
- Uses `cv2.minAreaRect()` / `boxPoints()` for final corner extraction (robust for rounded card corners).
- Returns card geometry used by `compute_scale_factor()`.

## Scale computation
- Physical reference: ISO ID-1 card size (`85.60mm x 53.98mm`).
- Output scale: `px_per_cm`.

## Important checks
- Candidate area sanity checks.
- Card aspect-ratio consistency near ID-1 ratio.
- Confidence score propagated to downstream confidence model.

## Debug
With `--debug`, intermediate images are written under `output/*card_detection_debug*/` (exact folder depends on output location).
