# Algorithms

Current, maintained algorithm documentation for the implemented pipeline.

## Pipeline (implemented)
1. Image quality checks (`src/image_quality.py`)
2. Credit card detection + scale (`src/card_detection.py`)
3. Hand/finger segmentation (`src/finger_segmentation.py`)
4. Axis + ring-zone localization (`src/geometry.py`)
5. Width measurement
   - contour method (`src/geometry.py`)
   - Sobel method (`src/edge_refinement.py`)
6. Confidence scoring (`src/confidence.py`)
7. Result/debug rendering (`src/debug_observer.py`)

## Algorithm docs in this folder
- `02-card-detection.md`
- `04-finger-segmentation.md`
- `05-landmark-axis.md`
- `07b-sobel-edge-refinement.md`

## Notes
- These docs describe the **current code behavior** (not historical design drafts).
- v0/v1 planning documents remain under `doc/v0` and `doc/v1`.
