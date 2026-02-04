# Ring Sizer v1: Landmark-Based Edge Refinement

## Overview

Version 1 enhances the finger measurement system with **Sobel edge refinement** for improved measurement accuracy. This eliminates the systematic ±0.5-2mm error from contour smoothing in v0, targeting <0.3mm mean absolute error.

**Key Innovation:** Replace mask contour-based width measurement with gradient-based edge detection using MediaPipe landmarks and bidirectional Sobel filtering.

---

## Documentation Structure

```
doc/v1/
├── README.md              # This file - v1 overview
├── PRD.md                 # Product Requirements Document
├── Plan.md                # Implementation Plan (6 phases)
├── Progress.md            # Progress log and status tracking
├── debug-output-guide.md  # Debug visualization guide (created during Phase 5)
└── algorithms/            # Algorithm documentation (created during Phase 6)
    ├── 05-landmark-axis.md
    └── 07b-sobel-edge-refinement.md
```

---

## What's New in v1

### Core Features

1. **Landmark-Based Axis Estimation**
   - Uses MediaPipe finger landmarks (MCP, PIP, DIP, TIP) directly for axis calculation
   - More robust than PCA for bent fingers
   - Falls back to PCA if landmarks unavailable

2. **Sobel Edge Refinement**
   - Bidirectional Sobel filtering perpendicular to finger axis
   - Detects pixel-precise edges instead of mask contours
   - Handles both brightness transitions (dark→bright, bright→dark)

3. **Sub-Pixel Edge Localization**
   - Parabola fitting on gradient magnitude
   - Achieves <0.5 pixel accuracy
   - ~0.1-0.2mm precision at typical resolutions

4. **Auto Fallback Logic**
   - Default mode tries Sobel first
   - Falls back to v0 contour method if edge quality low
   - Transparent reporting of method used

5. **Method Comparison Mode**
   - Side-by-side comparison of contour vs Sobel
   - Generates both measurements for validation
   - Visual comparison in debug output

### Expected Improvements

| Metric | v0 Baseline | v1 Target |
|--------|-------------|-----------|
| Mean Absolute Error | 0.8 mm | <0.3 mm |
| Standard Deviation | 0.5 mm | <0.2 mm |
| Edge Detection Success | 85% | >90% |

---

## Research Foundation

v1 is inspired by techniques from AR jewelry try-on research:

**Article:** [AR in Jewelry Retail - Jewelry Try-On](https://postindustria.com/ar-in-jewelry-retail-jewelry-tryon)

**Key Techniques Adopted:**
- Landmark-based finger axis (p13-p14 for ring finger)
- Bidirectional Sobel edge detection
- Perpendicular cross-section measurement

**Adaptations for Ring Sizing:**
- Sub-pixel precision for measurement accuracy
- Robust quality scoring and fallback logic
- Single-image processing (vs video stabilization)
- Credit card scale calibration integration

---

## Usage Examples

### Auto Mode (Recommended)
```bash
# Try Sobel, fall back to contour if needed
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --edge-method auto
```

### Force Sobel Method
```bash
# Use Sobel edge refinement (fail if detection fails)
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --edge-method sobel
```

### Comparison Mode
```bash
# Compare both methods side-by-side
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --edge-method compare \
  --debug output/debug.png
```

### v0 Compatibility Mode
```bash
# Use original contour method
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --edge-method contour
```

### Advanced Options
```bash
# Adjust Sobel parameters
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --edge-method sobel \
  --sobel-threshold 25 \
  --sobel-kernel-size 5 \
  --rotation-align
```

---

## JSON Output Format

### Basic Output (Compatible with v0)
```json
{
  "finger_outer_diameter_cm": 1.78,
  "confidence": 0.89,
  "edge_method_used": "sobel",
  "scale_px_per_cm": 42.3,
  "quality_flags": {
    "card_detected": true,
    "finger_detected": true,
    "view_angle_ok": true,
    "edge_quality_ok": true
  },
  "fail_reason": null
}
```

### Comparison Mode Output
```json
{
  "finger_outer_diameter_cm": 1.78,
  "confidence": 0.89,
  "edge_method_used": "compare",
  "method_comparison": {
    "contour_width_cm": 1.82,
    "sobel_width_cm": 1.78,
    "difference_cm": 0.04,
    "contour_confidence": 0.86,
    "sobel_confidence": 0.89
  },
  "scale_px_per_cm": 42.3,
  "quality_flags": {
    "card_detected": true,
    "finger_detected": true,
    "view_angle_ok": true,
    "edge_quality_ok": true
  },
  "fail_reason": null
}
```

---

## Debug Output

### Main Debug Overlay
Updated to include:
- Edge method indicator (contour/sobel)
- Edge quality score visualization
- Gradient strength heatmap
- Sub-pixel precision indicator

### Edge Refinement Debug Directory
New directory: `output/edge_refinement_debug/`

**15 images showing:**
1. Landmark axis overlay
2. Ring zone and ROI bounds
3. Extracted ROI (optionally rotated)
4. Sobel left-to-right gradient
5. Sobel right-to-left gradient
6. Combined gradient magnitude
7. Edge candidates above threshold
8. Selected left/right edges
9. Sub-pixel refinement visualization
10. Width measurement lines
11. Width distribution histogram
12. Outlier detection overlay
13. Contour vs Sobel comparison
14. Measurement difference heatmap
15. Confidence component breakdown

---

## Implementation Status

**Current Status:** Planning Complete ✅

**Next Steps:**
1. Phase 1: Landmark-based axis estimation (Week 1)
2. Phase 2: Sobel edge detection core (Week 2)
3. Phase 3: Sub-pixel refinement (Week 3)
4. Phase 4-5: Integration and debug visualization (Week 4)
5. Phase 6: Validation and documentation (Week 5)

See [Progress.md](Progress.md) for detailed implementation tracking.

---

## Backward Compatibility

**Guaranteed:**
- v0 JSON output format unchanged (new fields are optional)
- Existing CLI interface works unchanged
- Default behavior with no flags maintained
- v0 scripts continue to work

**New additions (non-breaking):**
- Optional `edge_method_used` field in JSON
- Optional `method_comparison` object in JSON
- New CLI flags (ignored by v0-aware scripts)
- New debug subdirectory (separate from main overlay)

---

## When to Use Each Method

### Auto Mode (Default)
- **Best for:** General use, production deployments
- **Behavior:** Tries Sobel, falls back to contour if quality low
- **Pros:** Optimal accuracy with reliability guarantee
- **Cons:** Non-deterministic method selection

### Sobel Mode
- **Best for:** High-quality images, controlled conditions
- **Behavior:** Always uses Sobel, fails if edge detection fails
- **Pros:** Maximum accuracy when it works
- **Cons:** May fail on challenging images

### Contour Mode
- **Best for:** Consistency with v0, debugging, comparison
- **Behavior:** Uses v0 contour method exclusively
- **Pros:** Proven reliability, consistent with historical data
- **Cons:** Lower accuracy than Sobel

### Compare Mode
- **Best for:** Validation, quality assessment, debugging
- **Behavior:** Runs both methods, generates comparison
- **Pros:** Full visibility into method differences
- **Cons:** Slower (2x processing), more debug output

---

## Configuration Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--edge-method` | `auto` | auto, contour, sobel, compare | Edge detection method |
| `--sobel-threshold` | 30.0 | 10-100 | Min gradient for valid edge |
| `--sobel-kernel-size` | 3 | 3, 5, 7 | Sobel kernel size |
| `--rotation-align` | False | - | Rotate ROI vertically |
| `--subpixel-precision` | True | - | Enable sub-pixel localization |

---

## Troubleshooting

### Sobel Edge Detection Fails

**Symptom:** Auto mode falls back to contour, or sobel mode fails

**Possible causes:**
- Uniform lighting (weak gradients)
- Textured background (noisy edges)
- Low resolution image
- Finger motion blur

**Solutions:**
1. Improve lighting (create stronger edges)
2. Use plain background
3. Increase image resolution
4. Try lower `--sobel-threshold` (e.g., 20)
5. Use contour mode for reliability

### Sobel vs Contour Disagree

**Symptom:** Compare mode shows >0.5cm difference

**Possible causes:**
- Contour smoothing artifact
- Edge detection in wrong location
- Scale calibration issue

**Solutions:**
1. Check debug images in `edge_refinement_debug/`
2. Look at `13_contour_vs_sobel.png` for visual comparison
3. Verify credit card detection quality
4. Check `edge_quality_ok` flag in output

### Performance Issues

**Symptom:** Processing takes >2 seconds

**Possible causes:**
- High resolution image
- Debug mode enabled
- Compare mode enabled

**Solutions:**
1. Disable debug mode for production
2. Use auto or sobel mode (not compare)
3. Resize input image to 1920px max dimension
4. Use faster Sobel kernel (size 3)

---

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Landmark axis estimation | <10ms | Fast, direct calculation |
| ROI extraction | <20ms | Depends on rotation |
| Sobel filtering | ~50ms | Depends on ROI size |
| Edge detection | ~30ms | 20 cross-sections |
| Sub-pixel refinement | ~20ms | Parabola fitting |
| **Total Sobel overhead** | **~130ms** | vs v0 contour method |
| **Total with debug** | **~200ms** | 15 debug images |

---

## Future Enhancements (v2+)

Potential improvements for future versions:

1. **Adaptive Sobel Parameters**
   - Automatically adjust thresholds based on image characteristics
   - Lighting-aware gradient thresholds

2. **Multi-Resolution Edge Detection**
   - Image pyramid approach for better edge localization
   - Coarse-to-fine refinement

3. **Machine Learning Edge Detection**
   - CNN-based edge detection for ultimate accuracy
   - Trained on finger edge dataset

4. **Video Support**
   - Multi-frame averaging for noise reduction
   - Temporal consistency checks
   - Homography-based stabilization

5. **Real-Time Optimization**
   - GPU acceleration
   - Optimized Sobel kernels
   - Streaming processing

---

## References

### Documentation
- [PRD.md](PRD.md) - Product requirements
- [Plan.md](Plan.md) - Implementation plan
- [Progress.md](Progress.md) - Progress tracking
- [doc/algorithms/README.md](../algorithms/README.md) - Algorithm index

### Research
- [AR in Jewelry Retail - Jewelry Try-On](https://postindustria.com/ar-in-jewelry-retail-jewelry-tryon) - Sobel edge refinement inspiration

### Related
- [doc/v0/README.md](../v0/README.md) - v0 documentation
- [CLAUDE.md](../../CLAUDE.md) - Development guidelines
- [README.md](../../README.md) - User guide
