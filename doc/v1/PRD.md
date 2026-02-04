# PRD: Landmark-Based Edge Refinement for Improved Measurement Accuracy (v1)

## 1. Purpose

Enhance the existing finger measurement system (v0) with **landmark-based edge refinement** to achieve sub-millimeter measurement accuracy. This builds upon the v0 foundation by replacing contour-based width measurement with pixel-precise edge detection using Sobel filtering and MediaPipe landmarks.

**Key Improvement**: Transition from mask-based contours to gradient-based edge detection, inspired by AR jewelry try-on research, to eliminate systematic measurement errors.

---

## 2. Context & Motivation

### 2.1 Current State (v0)

**v0 Measurement Pipeline:**
1. MediaPipe generates hand mask (pixel-level or polygon-based)
2. PCA estimates finger axis from entire mask
3. Cross-sections perpendicular to axis intersect with **contour extracted from mask**
4. Width measured as distance between contour intersections

**Limitations:**
- **Contour artifacts**: Morphological operations (closing, opening, blur) smooth edges, introducing ±0.5-2mm error
- **Mask quality dependency**: Width accuracy limited by mask generation quality
- **No sub-pixel precision**: Contour intersection operates at pixel level
- **Loss of edge information**: Real finger boundaries lost in mask processing

### 2.2 Research Insight

Article reference: [AR in Jewelry Retail - Jewelry Try-On](https://postindustria.com/ar-in-jewelry-retail-jewelry-tryon)

**Key Techniques from Article:**
1. **Landmark-based axis** - Use MediaPipe DIP-PIP joints (p13-p14 for ring finger) directly for finger axis
2. **Sobel edge refinement** - Apply horizontal Sobel kernels perpendicular to finger axis to find pixel-precise edges
3. **Bidirectional filtering** - Use both dark→bright and bright→dark kernels to handle gradient directionality
4. **Image rotation** - Optionally rotate finger to vertical alignment for consistent processing

**Applicability to v0:**
- ✅ Landmark-based axis - Already using landmarks for orientation, can make primary
- ✅ Sobel edge refinement - **Core innovation**, directly applicable
- ✅ Bidirectional filtering - Needed for robust edge detection
- ⚠️ Image rotation - Optional optimization, not essential for single-image measurement

### 2.3 Expected Impact

**Measurement Accuracy:**
- Current: ±0.5-2mm error from contour smoothing
- Target: <0.5mm error with sub-pixel edge detection
- Ring sizing impact: Each ring size ≈ 0.4mm, so 0.5mm = ~1 size difference

**Robustness:**
- Less sensitive to mask morphology parameters
- Direct edge detection more resilient to lighting variations
- Reduced dependency on mask generation quality

---

## 3. Scope (v1)

### In Scope

**Core Features:**
1. **Landmark-based axis estimation** - Use MediaPipe finger landmarks (MCP, PIP, DIP, TIP) as primary axis, PCA as fallback
2. **Sobel edge refinement** - Bidirectional Sobel filtering perpendicular to finger axis in ring zone
3. **Sub-pixel edge localization** - Gradient interpolation for sub-pixel precision
4. **Method comparison mode** - Side-by-side comparison of contour vs Sobel methods
5. **Enhanced confidence scoring** - Add edge quality metrics to confidence calculation

**Configuration:**
- `--edge-method` flag: `auto` (default), `contour`, `sobel`, `compare`
- Sobel parameters: kernel size, thresholds, gradient strength requirements
- Debug visualization showing edge detection process

**Compatibility:**
- Maintain backward compatibility with v0 JSON output format
- Add optional `edge_method_used` field to output
- Existing scripts continue to work unchanged

### Out of Scope (Future Versions)

- Image rotation preprocessing (can be added in v2 if needed)
- Video stabilization with homography (requires multi-frame input)
- Machine learning-based edge detection
- 3D depth-based measurements
- Real-time processing optimizations

---

## 4. Technical Approach

### 4.1 Landmark-Based Axis Estimation

**Current (v0):** PCA on entire finger mask → axis direction
**New (v1):** MediaPipe landmarks → direct axis calculation

**Implementation:**
```python
def estimate_finger_axis_from_landmarks(
    finger_landmarks: np.ndarray,  # 4x2 array: [MCP, PIP, DIP, TIP]
) -> Dict[str, Any]:
    """
    Calculate finger axis directly from anatomical landmarks.
    More robust than PCA for slightly bent fingers.
    """
    # Primary axis: MCP to TIP (full finger length)
    # Alternative: PIP to DIP (middle segment, more stable)

    # Use linear regression or median direction across segments
    # Fall back to PCA if landmarks unavailable or quality low
```

**Benefits:**
- Anatomically consistent across hand poses
- Not affected by mask artifacts at fingertip/palm edge
- Works better for bent fingers (PCA assumes straight finger)

**Ring Zone Definition:**
- Current: 15-25% of finger length from palm end
- Alternative: Relative to PIP joint landmark (more anatomically consistent)

### 4.2 Sobel Edge Refinement

**Pipeline:**

```
1. Define Ring Zone → 2. Extract ROI → 3. Rotate ROI (optional) →
4. Apply Sobel Filters → 5. Detect Edges → 6. Measure Width
```

**Stage 1: Ring Zone ROI Extraction**
- Define rectangular ROI perpendicular to finger axis within ring zone
- Expand ROI to include surrounding context (important for gradient calculation)
- Extract ROI from original image (RGB or grayscale)

**Stage 2: Optional Rotation Alignment**
- Rotate ROI so finger axis is vertical
- Simplifies Sobel kernel application (use horizontal kernels)
- Not strictly necessary but improves consistency

**Stage 3: Bidirectional Sobel Filtering**

**Left-to-right kernel (detects dark→bright, e.g., background to finger):**
```
[-1  0  +1]
[-2  0  +2]
[-1  0  +1]
```

**Right-to-left kernel (detects bright→dark, e.g., finger to background):**
```
[+1  0  -1]
[+2  0  +2]
[+1  0  -1]
```

Apply both kernels to capture edges regardless of brightness transition direction.

**Stage 4: Edge Detection**
- For each cross-section (row if vertically aligned):
  - Find local maxima in gradient magnitude
  - Threshold by gradient strength (reject weak edges)
  - Select leftmost and rightmost strong edges as finger boundaries

**Stage 5: Sub-Pixel Edge Localization**
- Fit parabola to gradient magnitude around detected edge pixel
- Find parabola peak for sub-pixel precision
- Achieves <0.5px accuracy (0.1-0.2mm at typical resolutions)

**Stage 6: Width Measurement**
- Compute width as distance between left and right edges
- Sample multiple cross-sections within ring zone (20 samples, same as v0)
- Use median width (robust to outliers)
- Convert to cm using scale factor from card detection

### 4.3 Method Comparison & Validation

**Comparison Mode (`--edge-method compare`):**
- Run both contour-based (v0) and Sobel-based (v1) methods
- Generate side-by-side debug visualization
- Output both measurements for validation
- Highlight differences and confidence scores

**Output Format:**
```json
{
  "finger_outer_diameter_cm": 1.78,
  "confidence": 0.89,
  "edge_method_used": "sobel",
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

## 5. Configuration & CLI Interface

### 5.1 New Command-Line Flags

```bash
# Use Sobel edge refinement (default: auto)
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --edge-method sobel

# Compare both methods
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --edge-method compare \
  --debug output/debug.png

# Adjust Sobel sensitivity
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --sobel-threshold 30 \
  --sobel-kernel-size 5
```

### 5.2 Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--edge-method` | `auto` | auto, contour, sobel, compare | Edge detection method |
| `--sobel-threshold` | 30 | 10-100 | Minimum gradient magnitude for valid edge |
| `--sobel-kernel-size` | 3 | 3, 5, 7 | Sobel kernel size (larger = smoother) |
| `--rotation-align` | False | - | Rotate ROI for vertical finger alignment |
| `--subpixel-precision` | True | - | Enable sub-pixel edge localization |

### 5.3 Auto Mode Behavior

When `--edge-method auto` (default):
1. Attempt Sobel edge refinement first
2. Check edge quality:
   - Sufficient gradient strength (>threshold)
   - Consistent edge detection across cross-sections (>80% success rate)
   - Edge positions reasonable (within finger mask bounds)
3. Fall back to contour method if Sobel fails quality checks
4. Report which method was used in `edge_method_used` field

---

## 6. Confidence Scoring Updates

### 6.1 New Confidence Component: Edge Quality

**v0 Confidence (3 components):**
- Card detection: 30%
- Finger detection: 30%
- Measurement stability: 40%

**v1 Confidence (4 components):**
- Card detection: 25%
- Finger detection: 25%
- **Edge quality: 20%** ← NEW
- Measurement stability: 30%

### 6.2 Edge Quality Metrics

For Sobel method:
- **Gradient strength** - Average magnitude of detected edges
- **Edge consistency** - % of cross-sections with valid edges
- **Edge smoothness** - Variance in edge positions along finger
- **Bilateral symmetry** - Left/right edge quality balance

For contour method (unchanged):
- Contour smoothness
- Intersection success rate

**Scoring:**
```
edge_quality_score = (
    0.4 * gradient_strength_score +
    0.3 * consistency_score +
    0.2 * smoothness_score +
    0.1 * symmetry_score
)
```

---

## 7. Debug Visualization Enhancements

### 7.1 New Debug Outputs

**Edge Refinement Debug Directory:** `output/edge_refinement_debug/`

**Images (15 stages):**

**Stage A: Axis & Zone (3 images)**
1. `01_landmark_axis.png` - Finger landmarks with axis overlay
2. `02_ring_zone_roi.png` - Ring zone highlighted, ROI bounds
3. `03_roi_extraction.png` - Extracted ROI image (rotated if enabled)

**Stage B: Sobel Filtering (5 images)**
4. `04_sobel_left.png` - Left-to-right gradient (dark→bright edges)
5. `05_sobel_right.png` - Right-to-left gradient (bright→dark edges)
6. `06_gradient_magnitude.png` - Combined gradient magnitude
7. `07_edge_candidates.png` - All pixels above gradient threshold
8. `08_selected_edges.png` - Final left/right edges per cross-section

**Stage C: Measurement (4 images)**
9. `09_subpixel_refinement.png` - Sub-pixel parabola fitting visualization
10. `10_width_measurements.png` - Width lines with measurements
11. `11_width_distribution.png` - Histogram of cross-section widths
12. `12_outlier_detection.png` - Highlighting outlier measurements

**Stage D: Comparison (3 images)**
13. `13_contour_vs_sobel.png` - Side-by-side edge comparison
14. `14_measurement_difference.png` - Heatmap of width differences
15. `15_confidence_comparison.png` - Component confidence breakdown

### 7.2 Main Debug Overlay Updates

Add to existing debug visualization:
- Edge method indicator (contour/sobel/compare)
- Edge quality score visualization
- Gradient strength heatmap along finger
- Sub-pixel precision indicator

---

## 8. Validation & Testing Strategy

### 8.1 Accuracy Validation

**Ground Truth Comparison:**
1. Collect test images with known finger widths (measured with calipers)
2. Compare v0 vs v1 measurements
3. Calculate Mean Absolute Error (MAE) and Standard Deviation

**Target Metrics:**
- MAE < 0.3mm (v0 baseline: ~0.8mm)
- Std Dev < 0.2mm (v0 baseline: ~0.5mm)
- Confidence correlation with actual error

### 8.2 Robustness Testing

**Test Conditions:**
- Lighting variations (bright, dim, mixed)
- Skin tones (diverse dataset)
- Finger positions (straight, slightly bent)
- Image quality (sharp, slight blur)
- Background textures (plain, patterned)

**Success Criteria:**
- Edge detection success rate >90% on valid images
- Auto fallback to contour works correctly
- No measurement outliers >2mm from ground truth

### 8.3 Performance Testing

**Timing Requirements:**
- Sobel edge refinement adds <200ms to pipeline (acceptable)
- Compare mode adds <300ms (generates both measurements)
- No increase in memory usage beyond v0

---

## 9. Migration & Backward Compatibility

### 9.1 Compatibility Guarantees

**Guaranteed to remain unchanged:**
- JSON output format (existing fields)
- CLI interface (existing flags)
- Default behavior with no flags
- Debug overlay filename convention

**New additions (non-breaking):**
- Optional `edge_method_used` field in JSON
- Optional `method_comparison` object in JSON
- New CLI flags (ignored by v0-aware scripts)
- New debug subdirectory (separate from main overlay)

### 9.2 Migration Path

**For users who want to stay on v0:**
```bash
# Explicit v0 behavior
python measure_finger.py --input img.jpg --output result.json --edge-method contour
```

**For users who want to try v1:**
```bash
# Use v1 with auto fallback
python measure_finger.py --input img.jpg --output result.json --edge-method auto

# Force v1 Sobel method
python measure_finger.py --input img.jpg --output result.json --edge-method sobel
```

**For validation:**
```bash
# Compare both methods
python measure_finger.py --input img.jpg --output result.json --edge-method compare --debug out.png
```

---

## 10. Failure Modes & Handling

### 10.1 New Failure Conditions

| Condition | Detection | Action |
|-----------|-----------|--------|
| Weak gradients (uniform lighting) | Avg gradient <10 | Fall back to contour |
| Noisy edges (textured background) | High edge variance | Increase threshold or fallback |
| No valid edges found | <50% cross-sections successful | Fall back to contour |
| Inconsistent edges (finger motion blur) | High position variance | Reduce confidence or fallback |
| Sobel kernel too large (low resolution) | ROI width < kernel * 5 | Use smaller kernel or fallback |

### 10.2 Graceful Degradation

**Fallback Cascade:**
1. Try Sobel with default parameters
2. If fails, try Sobel with relaxed threshold
3. If still fails, fall back to contour method (v0)
4. If contour fails, return measurement failure

**Error Reporting:**
- Set `edge_method_used: "contour_fallback"` when fallback occurs
- Add `edge_quality_flags` with failure reasons
- Reduce confidence score proportionally

---

## 11. Success Metrics

### 11.1 Quantitative Metrics

| Metric | v0 Baseline | v1 Target | Measurement Method |
|--------|-------------|-----------|-------------------|
| Mean Absolute Error | 0.8 mm | <0.3 mm | vs. caliper ground truth |
| Std Dev | 0.5 mm | <0.2 mm | Repeated measurements |
| Edge detection success | 85% | >90% | Auto mode, valid inputs |
| Processing time | 1.2s | <1.5s | Average per image |
| Confidence accuracy | 0.75 | >0.85 | Confidence vs actual error correlation |

### 11.2 Qualitative Goals

- Users report improved measurement consistency
- Fewer "outlier" measurements requiring recapture
- Better performance on challenging lighting conditions
- Debug visualizations clearly show edge quality
- Easy to understand when/why fallback occurs

---

## 12. Implementation Phases

### Phase 1: Foundation (Week 1)
- Implement landmark-based axis estimation
- Add axis visualization to debug output
- Validate axis accuracy vs PCA method

### Phase 2: Sobel Core (Week 2)
- Implement bidirectional Sobel filtering
- Basic edge detection without sub-pixel
- Generate initial debug visualizations

### Phase 3: Refinement (Week 3)
- Add sub-pixel edge localization
- Implement edge quality scoring
- Add auto fallback logic

### Phase 4: Integration (Week 4)
- Update confidence scoring
- Add method comparison mode
- Complete debug visualization suite
- Update documentation

### Phase 5: Validation (Week 5)
- Collect ground truth dataset
- Run accuracy validation tests
- Performance benchmarking
- Robustness testing

---

## 13. Documentation Requirements

### 13.1 User-Facing Documentation

**README.md updates:**
- Explain edge refinement feature
- When to use `--edge-method` flag
- Interpretation of edge quality scores
- Troubleshooting edge detection failures

**CLAUDE.md updates:**
- v1 implementation details
- Algorithm comparison table
- Parameter tuning guidelines

### 13.2 Technical Documentation

**doc/v1/algorithms/ (new):**
- `05-landmark-axis.md` - Landmark-based axis estimation
- `07b-sobel-edge-refinement.md` - Complete Sobel algorithm documentation
- Update algorithm README with v1 methods

**Debug Output Guide:**
- Document all 15 new debug images
- Interpretation guide for edge quality metrics
- Common failure patterns and solutions

---

## 14. Definition of Done (v1)

- [ ] Landmark-based axis estimation implemented and tested
- [ ] Bidirectional Sobel edge detection working
- [ ] Sub-pixel edge localization achieves <0.5px precision
- [ ] Auto fallback logic robust across test cases
- [ ] Edge quality confidence component integrated
- [ ] Method comparison mode generates side-by-side visualization
- [ ] All 15 edge refinement debug images generated
- [ ] JSON output includes `edge_method_used` field
- [ ] Accuracy validation shows MAE <0.3mm on test dataset
- [ ] Processing time <1.5s per image
- [ ] Edge detection success rate >90% on valid images
- [ ] Backward compatibility verified (v0 scripts still work)
- [ ] Documentation complete (README, CLAUDE.md, algorithm docs)
- [ ] Code reviewed and merged

---

## 15. Risk Assessment & Mitigation

### High-Risk Items

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Sobel method less accurate than expected | Medium | High | Maintain contour fallback, extensive testing |
| Performance regression | Low | Medium | Profile early, optimize hotspots, set time budget |
| Backward compatibility break | Low | High | Comprehensive integration testing, version flag |

### Medium-Risk Items

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Edge detection fails on certain skin tones | Medium | Medium | Diverse test dataset, adaptive thresholding |
| Complex debugging output confuses users | Medium | Low | Clear documentation, toggle debug modes |
| Increased maintenance burden | High | Low | Modular design, comprehensive tests |

---

## Appendix A: Research Article Summary

**Source:** [AR in Jewelry Retail - Jewelry Try-On](https://postindustria.com/ar-in-jewelry-retail-jewelry-tryon)

**Key Techniques Adopted:**
1. Landmark-based axis (p13-p14 for ring finger)
2. Sobel edge refinement (bidirectional kernels)
3. Perpendicular edge detection

**Key Techniques Deferred:**
4. Image rotation (optional optimization)
5. Homography stabilization (requires video)

**Why not 100% adoption:**
- Article targets AR try-on (4 points for ring positioning)
- We target precise measurement (sub-millimeter accuracy)
- Different use cases = different requirements
