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
1. Read doc/v0/PRD.md, doc/v0/Plan.md, doc/v0/Progress.md for baseline implementation
2. Read doc/v1/PRD.md, doc/v1/Plan.md, doc/v1/Progress.md for edge refinement (v1)
3. Assume this is a continuation of an existing project.
4. Summarize your understanding of the current state and propose the next concrete step without writing code yet.

## Project Overview

Ring Sizer is a **local, terminal-executable computer vision program** that measures the outer width (diameter) of a finger at the ring-wearing zone using a single RGB image. It uses a standard credit card (ISO/IEC 7810 ID-1: 85.60mm × 53.98mm) as a physical size reference for scale calibration.

**Key characteristics:**
- Single image input (JPG/PNG)
- **v1: Dual edge detection** - Landmark-based axis + Sobel gradient refinement
- MediaPipe-based hand and finger segmentation
- MediaPipe-based hand and finger segmentation
- Outputs JSON measurement data and optional debug visualization
- No cloud processing, runs entirely locally
- Python 3.8+ with OpenCV, NumPy, MediaPipe, and SciPy

## Development Commands

### Installation
```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Program
```bash
# Basic measurement (defaults to index finger, auto edge detection)
python measure_finger.py --input input/test_image.jpg --output output/result.json

# Measure specific finger (index, middle, ring, or auto)
python measure_finger.py \
  --input input/test_image.jpg \
  --output output/result.json \
  --finger-index ring

# With debug visualization
python measure_finger.py \
  --input input/test_image.jpg \
  --output output/result.json \
  --finger-index middle \
  --debug output/debug_overlay.png

# Force Sobel edge refinement (v1)
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --finger-index ring \
  --edge-method sobel \
  --sobel-threshold 15.0 \
  --debug output/debug.png

# Compare both methods
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --finger-index middle \
  --edge-method compare \
  --debug output/debug.png

# Force contour method (v0)
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --finger-index index \
  --edge-method contour
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

The codebase is organized into focused utility modules in `src/`:

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

---

## v1 Architecture (Edge Refinement)

### What's New in v1

v1 improves measurement accuracy by replacing contour-based edge detection with gradient-based Sobel edge refinement. Key improvements:

- **Landmark-based axis**: Uses MediaPipe finger landmarks (MCP→PIP→DIP→TIP) for more anatomically consistent axis estimation
- **Sobel edge detection**: Bidirectional gradient filtering for pixel-precise edge localization
- **Sub-pixel refinement**: Parabola fitting achieves <0.5px precision (~0.003cm at typical resolution)
- **Quality-based fallback**: Automatically uses v0 contour method if Sobel quality insufficient
- **Enhanced confidence**: Adds edge quality component (gradient strength, consistency, smoothness, symmetry)

### v1 Processing Pipeline (Enhanced Phases)

**Phase 5a: Landmark-Based Axis Estimation (v1)**
- Uses MediaPipe finger landmarks directly (4 points: MCP, PIP, DIP, TIP)
- **Finger selection**: Defaults to index finger, can specify middle or ring finger via `--finger-index`
- Orientation detection uses the **specified finger** for axis calculation (wrist → finger tip)
- Image automatically rotated to canonical orientation (wrist at bottom, fingers pointing up)
- Three axis calculation methods:
  - `endpoints`: Simple MCP→TIP vector
  - `linear_fit`: Linear regression on all 4 landmarks (default, most robust)
  - `median_direction`: Median of segment directions
- Falls back to PCA if landmarks unavailable or quality check fails
- Validation checks: NaN/inf, minimum spacing, monotonic progression, minimum length

**Phase 7b: Sobel Edge Refinement (v1)**
```
1. Extract ROI around ring zone → 2. Apply bidirectional Sobel filters →
3. Detect edges per cross-section → 4. Sub-pixel refinement → 5. Measure width
```

1. **ROI Extraction**
   - Rectangular region around ring zone with padding (50px for gradient context)
   - Width estimation: `finger_length / 3.0` (conservative)
   - Optional rotation alignment (not used by default)

2. **Bidirectional Sobel Filtering**
   - Applies `cv2.Sobel` with configurable kernel size (3, 5, or 7)
   - Computes gradient_x (horizontal edges), gradient_y (vertical edges)
   - Calculates gradient magnitude and direction
   - Auto-detects filter orientation from ROI aspect ratio

3. **Edge Detection Per Cross-Section**
   - **Mask-constrained mode** (primary):
     - Finds leftmost/rightmost finger mask pixels (finger boundaries)
     - Searches ±10px around boundaries for strongest gradient
     - Combines anatomical accuracy (mask) with sub-pixel precision (gradient)
   - **Gradient-only mode** (fallback): Pure Sobel without mask constraint

4. **Sub-Pixel Edge Localization**
   - Parabola fitting: f(x) = ax² + bx + c
   - Samples gradient at x-1, x, x+1
   - Finds parabola peak: x_peak = -b/(2a)
   - Constrains refinement to ±0.5 pixels
   - Achieves <0.5px precision (~0.003cm at 185 px/cm)

5. **Width Measurement**
   - Calculates width for each valid row
   - Outlier filtering using Median Absolute Deviation (MAD)
   - Removes measurements >3 MAD from median
   - Computes median, mean, std dev
   - Converts pixels to cm using scale factor

**Phase 8b: Enhanced Confidence Scoring (v1)**
- Adds 4th component: Edge Quality (20% weight)
  - Gradient strength: Avg magnitude at detected edges
  - Consistency: % of rows with valid edge pairs
  - Smoothness: Edge position variance (lower = better)
  - Symmetry: Left/right edge strength balance
- Reweights other components: Card 25%, Finger 25%, Measurement 30%

### v1 Module Structure

| Module | v1 Enhancements |
|--------|-----------------|
| `geometry.py` | Added `estimate_finger_axis_from_landmarks()`, `_validate_landmark_quality()`, landmark-based zone localization |
| **`edge_refinement.py`** | **[NEW]** Complete Sobel edge refinement pipeline with sub-pixel precision |
| `confidence.py` | Added `compute_edge_quality_confidence()`, dual-mode confidence calculation |
| `debug_observer.py` | Added 9 edge refinement drawing functions for visualization |
| `measure_finger.py` | CLI flags for edge method selection, method comparison mode |

### v1 CLI Flags

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--finger-index` | auto, index, middle, ring, pinky | **index** | Which finger to measure and use for orientation |
| `--edge-method` | auto, contour, sobel, compare | auto | Edge detection method |
| `--sobel-threshold` | float | 15.0 | Minimum gradient magnitude |
| `--sobel-kernel-size` | 3, 5, 7 | 3 | Sobel kernel size |
| `--no-subpixel` | flag | False | Disable sub-pixel refinement |

### v1 Auto Mode Behavior

When `--edge-method auto` (default):
1. Always computes contour measurement (v0 baseline)
2. Attempts Sobel edge refinement
3. Evaluates Sobel quality score (threshold: 0.7)
4. Checks consistency (>50% success rate required)
5. Verifies width reasonableness (0.8-3.5 cm)
6. Checks agreement with contour (<50% difference)
7. Uses Sobel if all checks pass, otherwise falls back to contour
8. Reports method used in `edge_method_used` field

### v1 Debug Output

When `--debug` flag used, generates:
- Main debug overlay (same as v0, shows final result)
- `output/edge_refinement_debug/` subdirectory with 12 images:
  - **Stage A** (3): Landmark axis, ring zone, ROI extraction
  - **Stage B** (5): Sobel gradients, candidates, selected edges
  - **Stage C** (4): Sub-pixel refinement, widths, distribution, outliers

### v1 Failure Modes (Additional)

- `sobel_edge_refinement_failed` - Sobel method explicitly requested but failed
- `quality_score_low_X.XX` - Edge quality below threshold (auto fallback)
- `consistency_low_X.XX` - Too few valid edge detections
- `width_unreasonable` - Measured width outside realistic range
- `disagreement_with_contour` - Sobel and contour differ by >50%

---

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
- **Orientation detection**: Uses wrist → specified finger tip to determine hand rotation
- **Automatic rotation**: Image rotated to canonical orientation (wrist at bottom, fingers up) based on selected finger

### Input Requirements
For optimal results:
- Resolution: 1080p or higher recommended
- View angle: Near top-down view
- **Finger**: One finger extended (index, middle, or ring). Specify with `--finger-index`
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
