# Gradient Filtering Techniques Guide

## Overview

Images 06a-06h show different image processing techniques applied to the gradient magnitude (06) to explore noise reduction and edge enhancement strategies.

**Format:** Each image shows side-by-side comparison:
- **Left**: Original gradient magnitude
- **Right**: Filtered gradient magnitude
- **Statistics**: Mean, Std, Max for both versions

---

## Filtering Techniques

### 06a: Gaussian Blur (5x5, sigma=1.5)
**Purpose:** Smooth out noise while preserving strong edges

**How it works:**
- Applies weighted average using Gaussian kernel
- Pixels farther from center have less influence
- Reduces high-frequency noise

**Effect on gradients:**
- Reduces noise spikes
- Slightly blurs edge boundaries
- Good for general smoothing

**Use case:** When you have uniform noise across the image

---

### 06b: Median Filter (5x5)
**Purpose:** Remove salt-and-pepper (impulse) noise

**How it works:**
- Replaces each pixel with median of neighborhood
- Non-linear filter (unlike Gaussian)
- Excellent at removing outliers

**Effect on gradients:**
- Removes isolated bright spots (noise spikes)
- Preserves edge sharpness better than Gaussian
- Eliminates extreme values

**Use case:** When you have scattered noise pixels (salt-and-pepper)

---

### 06c: Bilateral Filter (d=9, sigma=75)
**Purpose:** Edge-preserving smoothing

**How it works:**
- Combines spatial and intensity similarity
- Smooths within uniform regions
- Preserves edges where intensity changes sharply

**Effect on gradients:**
- Smooths flat regions (weak gradients)
- Preserves strong edge boundaries
- Best of both worlds: denoise + preserve edges

**Use case:** When you want to smooth noise but keep sharp edges

---

### 06d: Morphological Opening (3x3 ellipse)
**Purpose:** Remove small bright noise spots

**How it works:**
- Erosion followed by dilation
- Removes bright regions smaller than structuring element
- "Opens up" gaps between bright features

**Effect on gradients:**
- Removes small isolated bright pixels
- Slightly thins thick edges
- Cleans up scattered noise

**Use case:** When you have small bright noise spots to remove

---

### 06e: Morphological Closing (3x3 ellipse)
**Purpose:** Fill small dark gaps in edges

**How it works:**
- Dilation followed by erosion
- Fills dark regions smaller than structuring element
- "Closes" gaps in bright features

**Effect on gradients:**
- Fills small gaps in edge continuity
- Thickens thin edges slightly
- Makes edges more continuous

**Use case:** When edges have small breaks or gaps

---

### 06f: CLAHE (Contrast Limited Adaptive Histogram Equalization)
**Purpose:** Enhance local contrast adaptively

**How it works:**
- Divides image into tiles (8x8)
- Applies histogram equalization to each tile
- Limits contrast amplification (clip limit = 2.0)
- Blends tiles smoothly

**Effect on gradients:**
- Makes weak edges more visible
- Can amplify noise in uniform regions
- Enhances local details

**Use case:** When weak edges are hard to detect in low-contrast regions

**Warning:** Can amplify noise if applied to noisy images

---

### 06g: Non-Local Means Denoising (h=10)
**Purpose:** Advanced denoising while preserving structure

**How it works:**
- Compares patches (not just pixels) across entire image
- Similar patches are averaged together
- Preserves texture patterns and structures
- More computationally expensive

**Effect on gradients:**
- Excellent noise reduction
- Preserves edge structure
- Maintains texture patterns
- Cleaner than simple filters

**Use case:** When you want maximum noise reduction with minimal edge loss

**Note:** Slower than other filters (~100-200ms)

---

### 06h: Unsharp Masking (amount=1.5)
**Purpose:** Enhance edges by subtracting blur

**How it works:**
- Creates blurred version of image
- Subtracts blur from original: sharp = original - blur
- Adds sharp version back: enhanced = original + amount * sharp
- Formula: enhanced = original * 1.5 - blur * 0.5

**Effect on gradients:**
- Makes edges sharper and more prominent
- Increases gradient magnitude at boundaries
- Can amplify noise

**Use case:** When edges are too subtle and need enhancement

**Warning:** Amplifies noise along with edges

---

## Interpretation Guide

### Looking at Statistics

**Mean:**
- Higher mean = brighter overall (more strong gradients)
- Lower mean = darker overall (weaker gradients)

**Std (Standard Deviation):**
- Higher std = more variation (edges + noise)
- Lower std = more uniform (smoother)

**Max:**
- Shows strongest gradient in image
- High max = very strong edges present

### Comparing Filters

**For Noise Reduction (lower std, cleaner):**
1. Non-local means (06g) - Best quality, slowest
2. Bilateral (06c) - Good quality, moderate speed
3. Median (06b) - Good for impulse noise
4. Gaussian (06a) - Fast, general purpose
5. Morphological (06d/06e) - Specific noise patterns

**For Edge Enhancement (higher max, sharper):**
1. Unsharp masking (06h) - Strongest enhancement
2. CLAHE (06f) - Adaptive local enhancement
3. Morphological closing (06e) - Edge continuity

**For Edge Preservation:**
1. Bilateral (06c) - Best preservation during smoothing
2. Non-local means (06g) - Excellent structure preservation
3. Median (06b) - Good edge preservation

---

## Recommendations for Edge Detection

### Current Approach
Uses raw gradient magnitude (06) with threshold-based edge detection.

### Potential Improvements

**If seeing too much noise:**
1. Try bilateral filter (06c) first - good balance
2. If still noisy, try non-local means (06g)
3. For scattered spots, try median (06b)

**If missing weak edges:**
1. Try CLAHE (06f) to enhance local contrast
2. Try unsharp masking (06h) to sharpen edges
3. Lower gradient threshold

**If edges are broken:**
1. Try morphological closing (06e) to connect gaps
2. Use bilateral filter (06c) first to reduce noise

### Implementation Strategy

To integrate a filter into the pipeline:
```python
# In apply_sobel_filters(), after computing gradient_magnitude:
if use_preprocessing:
    gradient_magnitude = cv2.bilateralFilter(
        gradient_magnitude.astype(np.uint8), 9, 75, 75
    ).astype(np.float32)
```

### Testing Approach

1. Generate all 8 filter visualizations
2. Compare edge clarity and noise levels
3. Test edge detection with promising filters
4. Measure impact on:
   - Success rate (% valid edges detected)
   - Measurement accuracy (compare to ground truth)
   - Edge quality score

---

## Current Debug Image Sequence

```
Stage A: Axis & Zone
  01_landmark_axis.png
  02_ring_zone_roi.png
  03_roi_extraction.png

Stage B: Sobel Filtering
  04_sobel_left_to_right.png
  05_sobel_right_to_left.png
  06_gradient_magnitude.png
  06a_filter_gaussian.png          ← NEW
  06b_filter_median.png             ← NEW
  06c_filter_bilateral.png          ← NEW
  06d_filter_morph_open.png         ← NEW
  06e_filter_morph_close.png        ← NEW
  06f_filter_clahe.png              ← NEW
  06g_filter_nlm.png                ← NEW
  06h_filter_unsharp.png            ← NEW
  07a_all_candidates.png
  07b_filtered_candidates.png
  09_selected_edges.png

Stage C: Measurement
  10_subpixel_refinement.png
  11_width_measurements.png
  12_width_distribution.png
  13_outlier_detection.png
  14_comprehensive_overlay.png
```

---

## Next Steps

1. **Visual Review**: Examine all 8 filtered images
2. **Identify Best Filter**: Which shows clearest edges with least noise?
3. **A/B Testing**: Run edge detection with selected filter
4. **Measure Impact**: Compare success rate and accuracy
5. **Tune Parameters**: Adjust filter parameters if needed

