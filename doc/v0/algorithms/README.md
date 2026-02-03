# Algorithm Documentation

Detailed technical documentation for all algorithms in the Ring Sizer measurement system.

---

## 📋 Processing Pipeline Overview

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
        │  5. Finger Axis Estimation  │ ← PCA analysis
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  6. Ring Zone Localization  │ ← 15-25% from palm
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  7. Width Measurement       │ ← Cross-section sampling
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  8. Confidence Scoring      │ ← Multi-factor analysis
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
**Status:** 🔜 To be documented
**Module:** `src/geometry.py` (estimate_finger_axis)

- Principal Component Analysis (PCA)
- Primary axis calculation
- Orientation detection (palm vs tip)
- Finger length estimation
- Center point determination

**Document:** `05-axis-estimation.md` (coming soon)

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
**Status:** 🔜 To be documented
**Module:** `src/geometry.py` (compute_cross_section_width)

- 20 perpendicular cross-sections
- Line-contour intersection algorithm
- Edge detection (left/right)
- Median width calculation
- Pixel-to-centimeter conversion

**Document:** `07-width-measurement.md` (coming soon)

---

### Phase 8: Confidence Scoring
**Status:** 🔜 To be documented
**Module:** `src/confidence.py`

Multi-factor confidence assessment:
- **Card confidence** (30%): Detection quality, scale accuracy
- **Finger confidence** (30%): Landmark quality, mask validity
- **Measurement confidence** (40%): Width variance, outlier ratio

**Overall score:** Weighted average → HIGH/MEDIUM/LOW classification

**Document:** `08-confidence-scoring.md` (coming soon)

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

| Phase | Algorithm | Input | Output | Complexity |
|-------|-----------|-------|--------|------------|
| 1 | Image Quality | RGB Image | Quality Flags | O(n) |
| 2 | **Card Detection** ✅ | RGB Image | Corners, Confidence | O(n²) |
| 3 | Scale Calibration | Card Corners | px_per_cm | O(1) |
| 4 | **Finger Segment** ✅ | RGB Image | Mask, Landmarks | O(n) |
| 5 | Axis Estimation | Finger Mask | Axis, Center | O(n) |
| 6 | Zone Localization | Axis, Length | Zone Bounds | O(1) |
| 7 | Width Measurement | Zone, Scale | Width (cm) | O(n) |
| 8 | Confidence Scoring | All Phases | Confidence | O(1) |
| 9 | Visualization | All Results | Debug PNG | O(n) |

**Legend:**
- n = number of pixels in image
- ✅ = Documented
- 🔜 = To be documented

---

## 📖 Reading Guide

### For Algorithm Understanding
Read in sequential order:
1. Start with [02-card-detection.md](02-card-detection.md)
2. Continue with scale calibration (when available)
3. Follow the pipeline order above

### For Implementation
Focus on specific modules:
- **Detection:** 01, 02, 03
- **Segmentation:** 04, 05
- **Measurement:** 06, 07
- **Analysis:** 08, 09

### For Debugging
- **Card not detected:** See [02-card-detection.md](02-card-detection.md) - Strategy comparison
- **Poor measurements:** See 07-width-measurement.md (when available)
- **Low confidence:** See 08-confidence-scoring.md (when available)

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

**Last Updated:** 2026-02-03
**Documentation Version:** 1.0
