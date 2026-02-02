# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Standard Task Workflow

For tasks of implementing **new features**:
1. Read PRD.md, Plan.md, Progress.md before coding
2. Summarize current project state before implementation
3. Carry out the implementatation; after that, build and test if possible
4. Update Progress.md after changes
5. Commit with a clear, concise message

For tasks of **bug fixing**:
1. Summarize the bug, reason and solution before implementation
2. Carry out the implementation to fix the bug; build and test afterwards;
3. Update Progress.md after changes
4. Commit with a clear, concise message

For tasks of **reboot** from a new codex session:
1. Read PRD.md, Plan.md, Progress.md.
2. Assume this is a continuation of an existing project.
3. Summarize your understanding of the current state and propose the next concrete step without writing code yet.

## Project Overview

Ring Sizer is a **local, terminal-executable computer vision program** that measures the outer width (diameter) of a finger at the ring-wearing zone using a single RGB image. It uses a standard credit card (ISO/IEC 7810 ID-1: 85.60mm × 53.98mm) as a physical size reference for scale calibration.

**Key characteristics:**
- Single image input (JPG/PNG)
- MediaPipe-based hand and finger segmentation
- Outputs JSON measurement data and optional debug visualization
- No cloud processing, runs entirely locally
- Python 3.8+ with OpenCV, NumPy, MediaPipe, and SciPy

## Development Commands

### Installation
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Program
```bash
# Basic measurement
python measure_finger.py --input samples/test_image.jpg --output outputs/result.json

# With debug visualization
python measure_finger.py \
  --input samples/test_image.jpg \
  --output outputs/result.json \
  --debug outputs/debug_overlay.png

# Specify finger and confidence threshold
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --finger-index ring \
  --confidence-threshold 0.8 \
  --save-intermediate
```

## Architecture Overview

### Processing Pipeline (9 Phases)

The measurement pipeline follows a strict sequential flow:

1. **Image Quality Check** - Blur detection, exposure validation, resolution check
2. **Credit Card Detection & Scale Calibration** - Detects card, verifies aspect ratio (~1.586), computes `px_per_cm`
3. **Hand & Finger Segmentation** - MediaPipe hand detection, finger isolation, mask generation
4. **Finger Contour Extraction** - Extracts outer contour from cleaned mask
5. **Finger Axis Estimation** - PCA-based principal axis calculation, determines palm-end vs tip-end
6. **Ring-Wearing Zone Localization** - Defines zone at 15%-25% of finger length from palm-side
7. **Width Measurement** - Samples 20 cross-sections perpendicular to axis, uses median width
8. **Confidence Scoring** - Multi-factor scoring (card 30%, finger 30%, measurement 40%)
9. **Debug Visualization** - Generates annotated overlay image

### Module Structure

The codebase is organized into focused utility modules in `utils/`:

| Module | Primary Responsibilities |
|--------|--------------------------|
| `card_detection.py` | Credit card detection, perspective correction, scale calibration (`px_per_cm`) |
| `finger_segmentation.py` | MediaPipe integration, hand/finger isolation, mask cleaning, contour extraction |
| `geometry.py` | PCA axis estimation, ring zone localization, cross-section width measurement, line-contour intersections |
| `image_quality.py` | Blur detection (Laplacian variance), exposure checks, resolution validation |
| `confidence.py` | Component confidence scoring (card, finger, measurement), overall confidence computation |
| `visualization.py` | Debug overlay generation with contours, zones, measurements, and annotations |

### Key Design Decisions

**Ring-Wearing Zone Definition:**
- Located at 15%-25% of finger length from palm-side end
- Width measured by sampling 20 cross-sections within this zone
- Final measurement is the **median width** (robust to outliers)

**Axis Estimation:**
- Uses PCA (Principal Component Analysis) on finger mask points
- Determines palm-end vs tip-end using either:
  1. MediaPipe landmarks (preferred, if available)
  2. Thickness heuristic (thinner end is likely the tip)

**Confidence Scoring:**
- 3-component weighted average: Card (30%) + Finger (30%) + Measurement (40%)
- Confidence levels: HIGH (>0.85), MEDIUM (0.6-0.85), LOW (<0.6)
- Factors: card detection quality, finger mask area, width variance, aspect ratios

**Measurement Approach:**
- Perpendicular cross-sections to finger axis
- Line-contour intersection algorithm finds left/right edges
- Uses farthest pair of intersections to handle complex contours
- Converts pixels to cm using calibrated scale factor

## Important Technical Details

### What This Measures
The system measures the **external horizontal width** (outer diameter) of the finger at the ring-wearing zone. This is:
- ✅ The width of soft tissue + bone at the ring-wearing position
- ❌ NOT the inner diameter of a ring
- Used as a geometric proxy for downstream ring size mapping (out of scope for v0)

### Coordinate Systems
- Images use standard OpenCV format: (row, col) = (y, x)
- Most geometry functions work in (x, y) format
- Contours are Nx2 arrays in (x, y) format
- Careful conversion needed between formats (see `geometry.py:35`)

### MediaPipe Integration
- Uses pretrained hand landmark detection model (no custom training)
- Provides 21 hand landmarks per hand
- Each finger has 4 landmarks: MCP (base), PIP, DIP, TIP
- Finger indices: 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky

### Input Requirements
For optimal results:
- Resolution: 1080p or higher recommended
- View angle: Near top-down view
- Credit card: Must show at least 3 corners, aspect ratio ~1.586
- Finger and card must be on the same plane
- Good lighting, minimal blur

### Failure Modes
The system can fail at various stages:
- `card_not_detected` - Credit card not found or aspect ratio invalid
- `hand_not_detected` - No hand detected by MediaPipe
- `finger_isolation_failed` - Could not isolate specified finger
- `finger_mask_too_small` - Mask area too small after cleaning
- `contour_extraction_failed` - Could not extract valid contour
- `axis_estimation_failed` - PCA failed or insufficient points
- `zone_localization_failed` - Could not define ring zone
- `width_measurement_failed` - No valid cross-section intersections

## Output Format

### JSON Output Structure
```json
{
  "finger_outer_diameter_cm": 1.78,
  "confidence": 0.86,
  "scale_px_per_cm": 42.3,
  "quality_flags": {
    "card_detected": true,
    "finger_detected": true,
    "view_angle_ok": true
  },
  "fail_reason": null
}
```

### Debug Visualization Features
When `--debug` flag is used, generates an annotated image with:
- Credit card contour and corners (green)
- Finger contour (magenta, thick lines)
- Finger axis and endpoints (cyan/yellow)
- Ring-wearing zone band (yellow, semi-transparent)
- Cross-section sampling lines (orange)
- Measurement intersection points (blue circles)
- Final measurement and confidence text (large, readable font)

## Code Patterns and Conventions

### Error Handling
- Functions return `None` or raise exceptions on failure
- Main pipeline (`measure_finger()`) returns structured output dict with `fail_reason`
- Console logging provides detailed progress information

### Type Hints
- Extensive use of type hints throughout
- Dict return types with `Dict[str, Any]` for structured data
- NumPy arrays typed as `np.ndarray`
- Literal types for enums (e.g., `FingerIndex`)

### Data Flow
- All major functions return dictionaries with consistent keys
- Downstream functions accept upstream outputs directly
- Debug visualization receives all intermediate results
- Clean separation between detection, computation, and visualization

### Validation and Sanity Checks
- Finger width should be in realistic range: 1.0-3.0 cm (typical: 1.4-2.4 cm)
- Credit card aspect ratio should be close to 1.586
- View angle check: scale confidence should be >0.9 for accurate measurements
- Minimum mask area threshold prevents false detections
