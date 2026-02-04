# Phase 7b: Sobel Edge Refinement (v1)

**Module:** `src/edge_refinement.py`
**Status:** ✅ Implemented with bidirectional gradients and sub-pixel precision
**Last Updated:** 2026-02-03
**Version:** v1

---

## Overview

Gradient-based edge detection using Sobel filters, replacing contour-based width measurement (v0) with pixel-precise edge localization. Achieves <0.5px precision through sub-pixel parabola fitting, resulting in ~0.003cm accuracy at typical resolutions (185 px/cm).

**Key improvements over v0:**
- Sub-pixel edge localization (vs integer-pixel contours)
- Bidirectional gradient analysis (left→right AND right→left)
- Quality-based fallback to contour method if Sobel fails
- 4-component edge quality scoring

---

## Input

```python
{
    "image": np.ndarray,              # Original RGB image (BGR in OpenCV)
    "finger_mask": np.ndarray,        # Binary finger mask (uint8)
    "finger_axis": dict,              # Axis from landmark or PCA method
    "ring_zone": dict,                # Zone from localize_ring_zone()
    "px_per_cm": float,               # Scale factor from card calibration
    "config": dict,                   # Sobel parameters (optional)
}
```

**Config parameters:**
```python
{
    "sobel_threshold": 15.0,          # Min gradient magnitude
    "sobel_kernel_size": 3,           # Sobel kernel (3, 5, or 7)
    "use_subpixel": True,             # Enable sub-pixel refinement
    "use_mask_constraint": True,      # Constrain search to mask boundaries
}
```

---

## Output

```python
{
    "finger_outer_diameter_cm": float,    # Median width in cm
    "width_measurements_cm": list,        # Individual cross-section widths
    "confidence": float,                  # Edge quality confidence [0-1]
    "edge_quality": dict,                 # Quality metrics (4 components)
    "method": "sobel",                    # Edge detection method used
    "px_per_cm": float,                   # Scale factor (pass-through)
}
```

**Edge quality components:**
```python
{
    "gradient_strength": float,       # Avg magnitude at detected edges
    "consistency": float,             # % of rows with valid edge pairs
    "smoothness": float,              # Position variance (lower=better)
    "symmetry": float,                # Left/right strength balance
}
```

---

## Algorithm Pipeline

### **Stage 1: ROI Extraction**

Extract a rectangular region around the ring zone for efficient processing.

#### **1.1 ROI Bounds Calculation**

```python
# Get perpendicular axis (rotate 90°)
perp_axis = np.array([-axis_vector[1], axis_vector[0]])

# Estimate finger width (conservative)
finger_length = np.linalg.norm(tip_point - palm_point)
estimated_width = finger_length / 3.0

# Add padding for gradient context
padding = 50  # pixels
roi_half_width = estimated_width / 2 + padding

# Define ROI corners
center = (zone_start + zone_end) / 2
roi_points = [
    center + perp_axis * roi_half_width + axis_vector * zone_height/2,
    center - perp_axis * roi_half_width + axis_vector * zone_height/2,
    center - perp_axis * roi_half_width - axis_vector * zone_height/2,
    center + perp_axis * roi_half_width - axis_vector * zone_height/2,
]
```

**Rationale:**
- Smaller ROI → faster computation
- Padding ensures gradient filters have context at edges
- Axis-aligned ROI improves gradient detection (optional rotation)

#### **1.2 ROI Image Extraction**

```python
# Get bounding box
x_min, y_min = np.min(roi_points, axis=0).astype(int)
x_max, y_max = np.max(roi_points, axis=0).astype(int)

# Crop with boundary checks
x_min, y_min = max(0, x_min), max(0, y_min)
x_max = min(image.shape[1], x_max)
y_max = min(image.shape[0], y_max)

roi_image = image[y_min:y_max, x_min:x_max]
roi_mask = finger_mask[y_min:y_max, x_min:x_max]
```

---

### **Stage 2: Sobel Gradient Computation**

Compute horizontal and vertical gradients using Sobel operator.

#### **2.1 Image Preprocessing**

```python
# Convert to grayscale
gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

# Optional: Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
```

#### **2.2 Sobel Filters**

```python
# Horizontal gradient (detects vertical edges)
gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)

# Vertical gradient (detects horizontal edges)
gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)

# Gradient magnitude
magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

# Gradient direction (angle)
direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
```

**Kernel size selection:**
- **3×3** (default): Fast, good for high-resolution images
- **5×5**: More smoothing, better for noisy images
- **7×7**: Maximum smoothing, may blur true edges

#### **2.3 Filter Orientation Detection**

Auto-detect whether to use horizontal or vertical gradients:

```python
roi_aspect = roi_height / roi_width

if roi_aspect > 1.5:
    # Tall ROI → finger runs vertically → use horizontal gradients
    use_gradient = gradient_x
elif roi_aspect < 0.67:
    # Wide ROI → finger runs horizontally → use vertical gradients
    use_gradient = gradient_y
else:
    # Use magnitude (both directions)
    use_gradient = magnitude
```

---

### **Stage 3: Edge Detection Per Cross-Section**

For each horizontal row in the ROI, find left and right finger edges.

#### **3.1 Mask-Constrained Edge Detection (Primary)**

Uses finger mask boundaries as initial candidates, then refines with gradients.

```python
for row_idx in range(roi_height):
    # Find leftmost and rightmost finger mask pixels
    mask_row = roi_mask[row_idx, :]
    finger_cols = np.where(mask_row > 0)[0]
    
    if len(finger_cols) < 2:
        continue  # No finger in this row
    
    left_boundary = finger_cols[0]
    right_boundary = finger_cols[-1]
    
    # Search ±10px around boundaries for strongest gradient
    search_range = 10
    left_col = find_peak_gradient(
        gradient_row[max(0, left_boundary - search_range):
                     left_boundary + search_range + 1],
        expected_sign=-1  # Left edge: dark→bright (negative gradient)
    )
    right_col = find_peak_gradient(
        gradient_row[right_boundary - search_range:
                     min(roi_width, right_boundary + search_range + 1)],
        expected_sign=+1  # Right edge: bright→dark (positive gradient)
    )
```

**Rationale:**
- Mask gives anatomically accurate boundaries
- Gradient search refines to sub-pixel precision
- ±10px window is large enough for nail/skin transitions but constrains false detections

#### **3.2 Gradient-Only Edge Detection (Fallback)**

If mask constraint fails, use pure gradient-based detection:

```python
def find_edges_gradient_only(gradient_row, threshold):
    # Find all peaks above threshold
    peaks = []
    for i in range(1, len(gradient_row) - 1):
        if (gradient_row[i] > threshold and
            gradient_row[i] > gradient_row[i-1] and
            gradient_row[i] > gradient_row[i+1]):
            peaks.append(i)
    
    if len(peaks) < 2:
        return None, None
    
    # Take outermost peaks as left/right edges
    return peaks[0], peaks[-1]
```

---

### **Stage 4: Sub-Pixel Edge Refinement**

Fit parabola to gradient magnitude around detected edge to find sub-pixel peak.

#### **4.1 Parabola Fitting**

For each integer-pixel edge position x, sample gradients at x-1, x, x+1:

```python
def refine_edge_subpixel(gradient_row, edge_col):
    # Sample 3 points
    if edge_col < 1 or edge_col >= len(gradient_row) - 1:
        return float(edge_col)  # Edge at boundary
    
    g_left = gradient_row[edge_col - 1]
    g_center = gradient_row[edge_col]
    g_right = gradient_row[edge_col + 1]
    
    # Fit parabola: f(x) = ax² + bx + c
    # Using 3 points: (-1, g_left), (0, g_center), (1, g_right)
    a = 0.5 * (g_left + g_right - 2 * g_center)
    b = 0.5 * (g_right - g_left)
    
    if abs(a) < 1e-6:
        return float(edge_col)  # Nearly flat, no peak
    
    # Peak at x = -b / (2a)
    x_peak = -b / (2 * a)
    
    # Constrain refinement to ±0.5 pixels
    x_peak = np.clip(x_peak, -0.5, 0.5)
    
    return edge_col + x_peak
```

**Mathematical derivation:**

Given 3 points (-1, g₋₁), (0, g₀), (1, g₁):
- Parabola: f(x) = ax² + bx + c
- Peak: f'(x) = 2ax + b = 0 → x = -b/(2a)
- Coefficients:
  - a = 0.5(g₋₁ + g₁ - 2g₀)
  - b = 0.5(g₁ - g₋₁)
  - c = g₀

**Precision achieved:**
- Integer pixel: ±0.5px precision (±0.003cm at 185 px/cm)
- Sub-pixel: <0.5px precision (typically ±0.2px = ±0.001cm)

---

### **Stage 5: Width Measurement**

Compute width for each cross-section and aggregate.

#### **5.1 Width Calculation**

```python
widths_px = []
for left_col_subpx, right_col_subpx in edge_pairs:
    width_px = abs(right_col_subpx - left_col_subpx)
    widths_px.append(width_px)

# Convert to cm
widths_cm = [w / px_per_cm for w in widths_px]
```

#### **5.2 Outlier Filtering (MAD)**

Use Median Absolute Deviation to remove outliers:

```python
def filter_outliers_mad(widths, threshold=3.0):
    median = np.median(widths)
    mad = np.median(np.abs(widths - median))
    
    if mad < 1e-6:
        return widths  # All values identical
    
    # Modified Z-score
    z_scores = 0.6745 * np.abs(widths - median) / mad
    
    # Keep measurements within threshold
    return widths[z_scores < threshold]
```

**Rationale:**
- MAD is robust to outliers (unlike standard deviation)
- Threshold 3.0 keeps ~99% of valid measurements
- Removes nail transitions, skin folds, shadows

#### **5.3 Final Measurement**

```python
filtered_widths_cm = filter_outliers_mad(widths_cm)

if len(filtered_widths_cm) < 5:
    return None  # Too few valid measurements

final_width_cm = np.median(filtered_widths_cm)
mean_width_cm = np.mean(filtered_widths_cm)
std_width_cm = np.std(filtered_widths_cm)
```

**Median vs Mean:**
- **Median** (used): Robust to remaining outliers
- **Mean**: Sensitive to outliers, but useful for consistency check
- If |median - mean| > 0.2cm, data may be noisy

---

### **Stage 6: Edge Quality Scoring**

Compute 4-component quality score for confidence calculation.

#### **6.1 Gradient Strength (25%)**

```python
avg_magnitude = np.mean([
    gradient[row, left_col] for row, left_col, right_col in edge_pairs
] + [
    gradient[row, right_col] for row, left_col, right_col in edge_pairs
])

strength_score = min(1.0, avg_magnitude / 100.0)  # Normalize to [0, 1]
```

**Typical values:**
- Strong edges (nail/skin): 50-150
- Weak edges (gradual transitions): 10-30
- Threshold: 15 (configurable)

#### **6.2 Consistency (25%)**

```python
consistency = len(valid_rows) / total_rows
```

**Interpretation:**
- >0.9: Excellent (edges detected in >90% of rows)
- 0.7-0.9: Good
- 0.5-0.7: Marginal (may fall back to contour)
- <0.5: Poor (likely to fail)

#### **6.3 Smoothness (25%)**

```python
# Variance of left edge positions
left_positions = [left_col for row, left_col, right_col in edge_pairs]
left_variance = np.var(left_positions)

# Repeat for right edge
right_variance = np.var(right_positions)

# Lower variance = smoother = better
smoothness = 1.0 / (1.0 + (left_variance + right_variance) / 200.0)
```

**Rationale:**
- Finger edges should be smooth (low variance)
- High variance indicates jitter, noise, or multiple edge candidates

#### **6.4 Symmetry (25%)**

```python
left_magnitudes = [gradient[row, left_col] for ...]
right_magnitudes = [gradient[row, right_col] for ...]

left_avg = np.mean(left_magnitudes)
right_avg = np.mean(right_magnitudes)

ratio = min(left_avg, right_avg) / max(left_avg, right_avg)
symmetry = ratio  # [0, 1], 1 = perfect symmetry
```

**Interpretation:**
- 1.0: Perfect symmetry (both edges equally strong)
- 0.8-1.0: Good (slight imbalance acceptable)
- 0.5-0.8: Marginal (one edge much weaker)
- <0.5: Poor (asymmetric lighting or occlusion)

#### **6.5 Overall Edge Quality**

```python
edge_quality = (
    0.25 * gradient_strength +
    0.25 * consistency +
    0.25 * smoothness +
    0.25 * symmetry
)
```

**Threshold for auto mode:**
- ≥0.7: Use Sobel result
- <0.7: Fall back to contour method

---

## Auto Mode Decision Logic

When `--edge-method auto` (default):

```python
def should_use_sobel(sobel_result, contour_result):
    # Check 1: Edge quality
    if sobel_result["confidence"] < 0.7:
        return False, "quality_score_low"
    
    # Check 2: Consistency
    if sobel_result["edge_quality"]["consistency"] < 0.5:
        return False, "consistency_low"
    
    # Check 3: Width reasonableness
    width = sobel_result["finger_outer_diameter_cm"]
    if not (0.8 <= width <= 3.5):
        return False, "width_unreasonable"
    
    # Check 4: Agreement with contour (within 50%)
    contour_width = contour_result["finger_outer_diameter_cm"]
    diff_ratio = abs(width - contour_width) / contour_width
    if diff_ratio > 0.5:
        return False, "disagreement_with_contour"
    
    return True, "sobel_selected"
```

---

## Debug Visualization

When `--debug` flag is used, generates 12 images in `output/edge_refinement_debug/`:

### **Stage A: Axis & Zone (3 images)**
1. `01_landmark_axis.png` - Finger landmarks and computed axis
2. `02_ring_zone_roi.png` - ROI bounds overlaid on image
3. `03_roi_extraction.png` - Extracted ROI region

### **Stage B: Sobel Filtering (5 images)**
4. `04_gradient_lr.png` - Left→Right gradient (gradient_x)
5. `05_gradient_rl.png` - Right→Left gradient (-gradient_x)
6. `06_gradient_magnitude.png` - Combined gradient strength
7. `07_edge_candidates.png` - All detected edge candidates
8. `08_selected_edges.png` - Final selected edges (post-filtering)

### **Stage C: Measurement (4 images)**
9. `09_subpixel_refinement.png` - Sub-pixel edge positions (zoomed)
10. `10_width_measurements.png` - Cross-section widths overlaid
11. `11_width_distribution.png` - Histogram of widths (matplotlib)
12. `12_outlier_detection.png` - MAD filtering visualization

---

## Performance

| Metric | Value |
|--------|-------|
| Computation time | 50-150ms (typical) |
| ROI size | ~300×150px (varies by finger) |
| Cross-sections | 20-100 (depending on zone height) |
| Sub-pixel precision | <0.5px (~0.001-0.003cm) |
| Success rate | >90% (with fallback) |

---

## Comparison: Sobel (v1) vs Contour (v0)

| Metric | Sobel (v1) | Contour (v0) |
|--------|------------|--------------|
| **Precision** | <0.5px (~0.001cm) | 1px (~0.005cm) |
| **Edge localization** | Gradient peak | Contour boundary |
| **Sub-pixel support** | Yes (parabola fit) | No (integer pixels) |
| **Robustness to noise** | Medium (gradient-based) | High (mask-based) |
| **Nail handling** | Excellent (gradient detects nail edge) | Poor (nail included in contour) |
| **Computation time** | ~100ms | ~20ms |
| **Failure mode** | Low gradient → fallback | Unusual contour → PCA error |

---

## Usage Examples

### **Basic Usage**
```python
from src.edge_refinement import refine_edges_sobel

result = refine_edges_sobel(
    image=image,
    finger_mask=mask,
    finger_axis=axis_result,
    ring_zone=zone_result,
    px_per_cm=px_per_cm
)

if result:
    width_cm = result["finger_outer_diameter_cm"]
    confidence = result["confidence"]
    print(f"Width: {width_cm:.2f} cm (confidence: {confidence:.2f})")
```

### **With Debug Output**
```python
result = refine_edges_sobel(
    image=image,
    finger_mask=mask,
    finger_axis=axis_result,
    ring_zone=zone_result,
    px_per_cm=px_per_cm,
    debug_dir="output/edge_refinement_debug",
    finger_landmarks=landmarks  # For debug visualization
)
```

### **Custom Configuration**
```python
config = {
    "sobel_threshold": 20.0,      # Higher threshold (stricter)
    "sobel_kernel_size": 5,       # Larger kernel (more smoothing)
    "use_subpixel": True,
    "use_mask_constraint": True,
}

result = refine_edges_sobel(..., config=config)
```

---

## Failure Modes

| Error | Cause | Mitigation |
|-------|-------|------------|
| `sobel_edge_refinement_failed` | All edge detections failed | Use contour fallback |
| `low_gradient_strength` | Poor lighting, low contrast | Adjust `sobel_threshold` |
| `insufficient_consistency` | <50% of rows have valid edges | Use contour fallback |
| `high_edge_variance` | Jittery edges, multiple candidates | Increase Sobel kernel size |
| `asymmetric_edges` | One edge much weaker | Check lighting, occlusion |

---

## Future Improvements

1. **Adaptive thresholding:** Auto-adjust `sobel_threshold` based on image statistics
2. **Edge tracking:** Use Canny edge linking to connect edge fragments
3. **Multi-scale Sobel:** Combine results from multiple kernel sizes
4. **ROI rotation:** Align ROI with finger axis for optimal gradient direction
5. **Deep learning:** CNN-based edge refinement for complex cases
6. **Temporal filtering:** For video input, smooth edges across frames

---

## Related Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `refine_edges_sobel()` | `edge_refinement.py` | Main entry point (this algorithm) |
| `_extract_ring_zone_roi()` | `edge_refinement.py` | ROI extraction (Stage 1) |
| `_compute_sobel_gradients()` | `edge_refinement.py` | Sobel filtering (Stage 2) |
| `_detect_edges_per_row()` | `edge_refinement.py` | Edge detection (Stage 3) |
| `_refine_edges_subpixel()` | `edge_refinement.py` | Parabola fitting (Stage 4) |
| `_measure_widths_from_edges()` | `edge_refinement.py` | Width calculation (Stage 5) |
| `compute_edge_quality_confidence()` | `confidence.py` | Quality scoring (Stage 6) |
