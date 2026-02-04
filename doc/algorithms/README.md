# Algorithm Documentation

Detailed technical documentation for all algorithms in the Ring Sizer measurement system.

---

## 📋 Processing Pipeline Overview

### v0 Pipeline (Contour-Based)
```
┌─────────────────────────────────────────────────────────────┐
│                    Input Image (RGB)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  1. Image Quality Check     │
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  2. Card Detection          │ ← Multi-strategy approach
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  3. Scale Calibration       │ ← px_per_cm calculation
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  4. Hand & Finger Segment   │ ← MediaPipe landmarks
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  5. Finger Axis (PCA)       │ ← Principal component
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  6. Ring Zone Localization  │ ← 15-25% from palm
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  7. Width Measurement       │ ← Contour intersection
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  8. Confidence Scoring      │ ← 3-component (30/30/40)
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  9. Debug Visualization     │ ← Optional overlay
        └──────────────┬──────────────┘
                       │
            ┌──────────┴──────────┐
            │   JSON Output +     │
            │   Debug Image       │
            └─────────────────────┘
```

### v1 Pipeline (Sobel-Based, with Auto-Fallback)
```
┌─────────────────────────────────────────────────────────────┐
│                    Input Image (RGB)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  1-3. Quality, Card, Scale  │ ← Same as v0
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  4. Hand & Finger Segment   │ ← MediaPipe landmarks
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  5a. Axis (Landmarks)  ✨   │ ← MCP→PIP→DIP→TIP
        │      (fallback: PCA)        │
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  6. Ring Zone Localization  │ ← 15-25% from palm
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────────────┐
        │      7. Width Measurement            │
        │   ┌──────────┴──────────┐           │
        │   │   7a. Contour        │ ← v0     │
        │   │   (always computed)  │           │
        │   └──────────┬───────────┘           │
        │              │                        │
        │   ┌──────────┴───────────┐           │
        │   │   7b. Sobel  ✨      │ ← v1     │
        │   │   (sub-pixel edges)  │           │
        │   └──────────┬───────────┘           │
        │              │                        │
        │   ┌──────────┴───────────┐           │
        │   │  Quality Check       │           │
        │   │  (threshold: 0.7)    │           │
        │   └──────────┬───────────┘           │
        │              │                        │
        │   ┌──────────┴───────────┐           │
        │   │  Select Best Method  │           │
        │   │  (Sobel or Contour)  │           │
        │   └──────────┬───────────┘           │
        └──────────────┬──────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  8b. Enhanced Confidence    │ ← 4-component (25/25/20/30)
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  9. Debug Visualization     │ ← 13 debug images
        └──────────────┬──────────────┘
                       │
            ┌──────────┴──────────┐
            │   JSON Output +     │
            │   Debug Images      │
            └─────────────────────┘
```

**Legend:**
- ✨ = v1 Enhancement
- v0 = Original contour-based method
- v1 = New Sobel-based method with auto-fallback

---

## 📚 Algorithm Documentation

### Phase 1: Image Quality Assessment
**Status:** 🔜 To be documented
**Module:** `src/image_quality.py`

- Blur detection (Laplacian variance)
- Exposure validation
- Resolution checks
- Early exit for poor quality images

**Document:** `01-image-quality.md` (coming soon)

---

### Phase 2: Credit Card Detection ✅
**Status:** ✅ **Documented**
**Module:** `src/card_detection.py`

Multi-strategy detection approach using 4 parallel algorithms:
1. **Canny Edge Detection** - High contrast edges
2. **Adaptive Thresholding** - Varying lighting
3. **Otsu's Thresholding** - Automatic threshold
4. **Color-Based Segmentation** - HSV gray detection

**Candidate scoring:**
- Area ratio validation (1-50% of image)
- Aspect ratio check (1.586 ± 15%)
- Corner angle verification (90° ± 25°)
- Weighted scoring: 40% area + 30% ratio + 30% angle

**Document:** **[02-card-detection.md](02-card-detection.md)** ✅

---

### Phase 3: Scale Calibration
**Status:** 🔜 To be documented
**Module:** `src/card_detection.py` (compute_scale_factor)

- Perspective correction of detected card
- Physical dimension mapping (85.60 × 53.98 mm)
- Pixels-per-centimeter calculation
- Calibration confidence estimation

**Document:** `03-scale-calibration.md` (coming soon)

---

### Phase 4: Hand & Finger Segmentation ✅
**Status:** ✅ **Documented**
**Module:** `src/finger_segmentation.py`

Dual-method approach for finger isolation:
1. **Pixel-Level Segmentation** (primary) - Preserves actual MediaPipe edges
2. **Polygon-Based Segmentation** (fallback) - Synthetic geometric approximation

**Key features:**
- MediaPipe 21-point hand landmark detection
- Multi-rotation detection (0°, 90°, 180°, 270°)
- Pixel-accurate hand mask generation
- Automatic finger selection by extension score
- ROI-based finger isolation with component analysis
- Morphological mask cleaning (7x7 kernel)
- Contour extraction with smoothing

**Accuracy improvement:** +25% width measurement (pixel-level vs polygon)

**Document:** **[04-finger-segmentation.md](04-finger-segmentation.md)** ✅

---

### Phase 5: Finger Axis Estimation
**Status:** 🔜 To be documented (v0 PCA method)
**Module:** `src/geometry.py` (estimate_finger_axis)

- Principal Component Analysis (PCA)
- Primary axis calculation
- Orientation detection (palm vs tip)
- Finger length estimation
- Center point determination

**Document:** `05-axis-estimation.md` (coming soon)

---

### Phase 5a: Landmark-Based Finger Axis (v1) ✅
**Status:** ✅ **Documented**
**Module:** `src/geometry.py` (estimate_finger_axis_from_landmarks)
**Version:** v1

Enhanced axis estimation using MediaPipe finger landmarks:
1. **3 Calculation Methods:**
   - Endpoints: Simple MCP→TIP vector
   - Linear Fit: Regression on 4 landmarks (default)
   - Median Direction: Robust to outliers
2. **Quality Validation:** NaN checks, spacing, monotonic progression
3. **Auto-Fallback:** Falls back to PCA if landmarks fail

**Accuracy:** More anatomically consistent than PCA

**Document:** **[05-landmark-axis.md](05-landmark-axis.md)** ✅

---

### Phase 6: Ring Zone Localization
**Status:** 🔜 To be documented
**Module:** `src/geometry.py` (localize_ring_zone)

- Zone definition: 15-25% from palm-side end
- Projection onto finger axis
- Start/end point calculation
- Validation and fallback strategies

**Document:** `06-zone-localization.md` (coming soon)

---

### Phase 7: Width Measurement
**Status:** 🔜 To be documented (v0 contour method)
**Module:** `src/geometry.py` (compute_cross_section_width)

- 20 perpendicular cross-sections
- Line-contour intersection algorithm
- Edge detection (left/right)
- Median width calculation
- Pixel-to-centimeter conversion

**Document:** `07-width-measurement.md` (coming soon)

---

### Phase 7b: Sobel Edge Refinement (v1) ✅
**Status:** ✅ **Documented**
**Module:** `src/edge_refinement.py` (refine_edges_sobel)
**Version:** v1

Gradient-based edge detection with sub-pixel precision:
1. **ROI Extraction:** Localized region around ring zone
2. **Sobel Filtering:** Bidirectional gradients (L→R, R→L)
3. **Edge Detection:** Mask-constrained gradient peak search
4. **Sub-Pixel Refinement:** Parabola fitting (<0.5px precision)
5. **Quality Scoring:** 4 metrics (strength, consistency, smoothness, symmetry)

**Precision:** <0.5px (~0.001-0.003cm at 185 px/cm)
**Auto-Fallback:** Falls back to contour if edge quality <0.7

**Document:** **[07b-sobel-edge-refinement.md](07b-sobel-edge-refinement.md)** ✅

---

### Phase 8: Confidence Scoring
**Status:** 🔜 To be documented (v0 3-component)
**Module:** `src/confidence.py`

Multi-factor confidence assessment (v0):
- **Card confidence** (30%): Detection quality, scale accuracy
- **Finger confidence** (30%): Landmark quality, mask validity
- **Measurement confidence** (40%): Width variance, outlier ratio

**Overall score:** Weighted average → HIGH/MEDIUM/LOW classification

**Document:** `08-confidence-scoring.md` (coming soon)

---

### Phase 8b: Enhanced Confidence Scoring (v1)
**Status:** 🔜 To be documented
**Module:** `src/confidence.py` (compute_edge_quality_confidence)
**Version:** v1

Enhanced confidence with edge quality component (v1):
- **Card confidence** (25%): Detection quality, scale accuracy
- **Finger confidence** (25%): Landmark quality, mask validity
- **Edge quality** (20%): Gradient strength, consistency, smoothness, symmetry
- **Measurement confidence** (30%): Width variance, outlier ratio

**Overall score:** Weighted average → HIGH/MEDIUM/LOW classification

**Document:** `08b-enhanced-confidence.md` (coming soon)

---

### Phase 9: Debug Visualization
**Status:** 🔜 To be documented
**Module:** `src/visualization.py`

- Overlay generation
- Card contour and corners (green)
- Finger contour (magenta)
- Axis and endpoints (cyan/yellow)
- Ring zone band (yellow transparent)
- Cross-sections and measurements
- Result annotations

**Document:** `09-visualization.md` (coming soon)

---

## 🔍 Quick Reference Table

### v0 Pipeline (Contour-Based)

| Phase | Algorithm | Input | Output | Complexity |
|-------|-----------|-------|--------|------------|
| 1 | Image Quality | RGB Image | Quality Flags | O(n) |
| 2 | **Card Detection** ✅ | RGB Image | Corners, Confidence | O(n²) |
| 3 | Scale Calibration | Card Corners | px_per_cm | O(1) |
| 4 | **Finger Segment** ✅ | RGB Image | Mask, Landmarks | O(n) |
| 5 | Axis Estimation (PCA) | Finger Mask | Axis, Center | O(n) |
| 6 | Zone Localization | Axis, Length | Zone Bounds | O(1) |
| 7 | Width Measurement (Contour) | Zone, Scale | Width (cm) | O(n) |
| 8 | Confidence Scoring (3-comp) | All Phases | Confidence | O(1) |
| 9 | Visualization | All Results | Debug PNG | O(n) |

### v1 Pipeline (Sobel-Based)

| Phase | Algorithm | Input | Output | Complexity |
|-------|-----------|-------|--------|------------|
| 1-4 | (Same as v0) | | | |
| 5a | **Axis (Landmarks)** ✅ | MediaPipe Landmarks | Axis, Endpoints | O(1) |
| 6 | Zone Localization | Axis, Length | Zone Bounds | O(1) |
| 7b | **Sobel Edge Refinement** ✅ | Zone ROI, Mask | Width (cm), Quality | O(n·m) |
| 8b | Enhanced Confidence (4-comp) | All Phases + Edge | Confidence | O(1) |
| 9 | Visualization (Enhanced) | All Results | 13 Debug PNGs | O(n) |

**Notes:**
- n = number of pixels in image
- m = number of cross-sections in zone (~20-100)
- ✅ = Documented
- 🔜 = To be documented

---

## 📖 Reading Guide

### For Algorithm Understanding
**v0 (Contour-Based):** Read in sequential order:
1. Start with [02-card-detection.md](02-card-detection.md)
2. Continue with [04-finger-segmentation.md](04-finger-segmentation.md)
3. Follow phases 5-9 (when available)

**v1 (Sobel-Based):** After understanding v0:
1. Read [05-landmark-axis.md](05-landmark-axis.md) - Enhanced axis estimation
2. Read [07b-sobel-edge-refinement.md](07b-sobel-edge-refinement.md) - Sub-pixel edges
3. Understand auto-fallback logic and quality scoring

### For Implementation
Focus on specific modules:
- **Detection:** 01, 02, 03
- **Segmentation:** 04, 05, 05a
- **Measurement (v0):** 06, 07
- **Measurement (v1):** 06, 05a, 07b
- **Analysis:** 08, 08b, 09

### For Debugging
- **Card not detected:** See [02-card-detection.md](02-card-detection.md) - Strategy comparison
- **Hand/finger not detected:** See [04-finger-segmentation.md](04-finger-segmentation.md) - Multi-rotation
- **Axis estimation failed:** See [05-landmark-axis.md](05-landmark-axis.md) - Quality validation
- **Edge refinement failed:** See [07b-sobel-edge-refinement.md](07b-sobel-edge-refinement.md) - Quality scoring
- **Poor measurements:** Compare v0 contour vs v1 Sobel methods
- **Low confidence:** See 08-confidence-scoring.md / 08b-enhanced-confidence.md (when available)

---

## 🔗 Related Documentation

- **[PRD.md](../PRD.md)** - Product requirements and specifications
- **[Plan.md](../Plan.md)** - Implementation plan and phasing
- **[Progress.md](../Progress.md)** - Development progress log
- **[CLAUDE.md](../../../CLAUDE.md)** - AI assistant guidance

---

## 📝 Documentation Standards

Each algorithm document should include:

1. **Overview** - Purpose and approach
2. **Algorithm Details** - Step-by-step pseudocode
3. **Parameters** - All constants and thresholds
4. **Strengths & Weaknesses** - When it works/fails
5. **Debug Output** - File mappings and visualization
6. **Examples** - Worked examples with numbers
7. **Related Docs** - Cross-references

---

## 🚀 Contributing

When documenting a new algorithm:

1. Copy the structure from `02-card-detection.md`
2. Use consistent formatting and code blocks
3. Include visual diagrams where helpful
4. Add examples and edge cases
5. Update this README with links
6. Cross-reference related algorithms

---

**Last Updated:** 2026-02-03 (v1 documentation added)
**Documentation Version:** 2.0 (v0 + v1)
