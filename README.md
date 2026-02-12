---
title: Ring Sizer
emoji: "\U0001F48D"
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Ring Sizer

Local computer-vision CLI tool that measures **finger outer diameter** from a single image using a **credit card** as scale reference.

## Live Demo
- Hugging Face Space: [https://huggingface.co/spaces/feng-x/ring-sizer](https://huggingface.co/spaces/feng-x/ring-sizer)
- Anyone can try the hosted web demo directly in the browser.

## What it does
- Detects a credit card and computes `px/cm` scale.
- Detects hand/finger with MediaPipe.
- Measures finger width in the ring-wearing zone.
- Supports dual edge modes:
  - `contour` (v0 baseline)
  - `sobel` (v1 refinement)
  - `auto` (default, Sobel with quality fallback)
  - `compare` (returns both method stats)
- Writes JSON output and always writes a result PNG next to it.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python measure_finger.py --input input/test_image.jpg --output output/result.json
```

### Common options
```bash
# Enable intermediate debug folders (card/finger/edge stages)
python measure_finger.py --input image.jpg --output output/result.json --debug

# Finger selection
python measure_finger.py --input image.jpg --output output/result.json --finger-index ring

# Force method
python measure_finger.py --input image.jpg --output output/result.json --edge-method contour
python measure_finger.py --input image.jpg --output output/result.json --edge-method sobel

# Compare contour vs sobel
python measure_finger.py --input image.jpg --output output/result.json --edge-method compare

# Sobel tuning
python measure_finger.py --input image.jpg --output output/result.json \
  --edge-method sobel --sobel-threshold 15 --sobel-kernel-size 3 --no-subpixel
```

## CLI flags (current)
- `--input` (required)
- `--output` (required)
- `--debug` (boolean; saves intermediate debug folders)
- `--save-intermediate`
- `--finger-index {auto,index,middle,ring,pinky}` (default `index`)
- `--confidence-threshold` (default `0.7`)
- `--edge-method {auto,contour,sobel,compare}` (default `auto`)
- `--sobel-threshold` (default `15.0`)
- `--sobel-kernel-size {3,5,7}` (default `3`)
- `--no-subpixel`
- `--skip-card-detection` (testing only)

## Output JSON
```json
{
  "finger_outer_diameter_cm": 1.78,
  "confidence": 0.91,
  "scale_px_per_cm": 203.46,
  "quality_flags": {
    "card_detected": true,
    "finger_detected": true,
    "view_angle_ok": true
  },
  "fail_reason": null,
  "edge_method_used": "contour_fallback",
  "method_comparison": {
    "contour": {
      "width_cm": 1.82,
      "width_px": 371.2,
      "std_dev_px": 3.8,
      "coefficient_variation": 0.01,
      "num_samples": 20,
      "method": "contour"
    },
    "sobel": {
      "width_cm": 1.78,
      "width_px": 362.0,
      "std_dev_px": 3.1,
      "coefficient_variation": 0.008,
      "num_samples": 140,
      "subpixel_used": true,
      "success_rate": 0.42,
      "edge_quality_score": 0.81,
      "method": "sobel"
    },
    "difference": {
      "absolute_cm": -0.04,
      "absolute_px": -9.2,
      "relative_pct": -2.2,
      "precision_improvement": 0.7
    },
    "recommendation": {
      "use_sobel": true,
      "reason": "quality_acceptable",
      "preferred_method": "sobel"
    },
    "quality_comparison": {
      "contour_cv": 0.01,
      "sobel_cv": 0.008,
      "sobel_quality_score": 0.81,
      "sobel_gradient_strength": 0.82,
      "sobel_consistency": 0.42,
      "sobel_smoothness": 0.91,
      "sobel_symmetry": 0.95
    }
  }
}
```

Notes:
- `edge_method_used` and `method_comparison` are optional (present when relevant).
- Result image path is auto-derived: `output/result.json` -> `output/result.png`.

## Documentation map
- Requirement docs: `doc/v{i}/PRD.md`, `doc/v{i}/Plan.md`, `doc/v{i}/Progress.md`
- Algorithms index: `doc/algorithms/README.md`
- Scripts: `script/README.md`
- Web demo: `web_demo/README.md`
