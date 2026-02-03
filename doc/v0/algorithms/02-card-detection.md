# Credit Card Detection

The credit card detection system uses a **multi-strategy approach** to robustly detect credit cards under varying conditions (lighting, orientation, background, camera settings). All strategies run in parallel and their results are combined for final selection.

**Module:** `src/card_detection.py`

## Overview

```
Input Image
    ↓
Preprocessing (Grayscale + Bilateral Filter)
    ↓
┌─────────────────────────────────────────────┐
│  4 Parallel Detection Strategies            │
├─────────────────────────────────────────────┤
│  1. Canny Edge Detection                    │
│  2. Adaptive Thresholding                   │
│  3. Otsu's Thresholding                     │
│  4. Color-Based Segmentation                │
└─────────────────────────────────────────────┘
    ↓
Candidate Pool (all detected quadrilaterals)
    ↓
Scoring & Filtering
    ↓
Best Candidate Selection
```

---

## Multi-Strategy Detection

### Preprocessing

Before running detection strategies, the image undergoes preprocessing:

1. **Grayscale Conversion**: `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`
2. **Bilateral Filtering**: `cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)`
   - Smooths image while preserving edges
   - Reduces noise without blurring card boundaries

**Debug Output:**
- `01_original.png` - Original input image
- `02_grayscale.png` - Grayscale conversion
- `03_bilateral_filtered.png` - Filtered result

---

### Strategy 1: Canny Edge Detection

**Purpose:** Detect card by finding strong edges that form quadrilaterals.

**Approach:**
- Uses 5 different threshold combinations to handle varying edge strengths
- Applies morphological closing to connect broken edges
- Extracts quadrilaterals from edge contours

**Algorithm:**
```python
for (low, high) in [(20,60), (30,100), (50,150), (75,200), (100,250)]:
    # 1. Detect edges
    edges = cv2.Canny(filtered, low, high)

    # 2. Connect broken edges with morphological closing
    kernel = 5×5 rectangle
    edges_morphed = cv2.morphologyEx(edges, MORPH_CLOSE, kernel)

    # 3. Find contours
    contours = cv2.findContours(edges_morphed, RETR_LIST, CHAIN_APPROX_SIMPLE)

    # 4. Extract quadrilaterals
    quads = approximate_to_quads(contours)
    candidates.extend(quads)
```

**Parameters:**
- **Canny Thresholds**: 5 configurations from sensitive (20,60) to strict (100,250)
- **Morphology Kernel**: 5×5 rectangle for edge connection
- **Approximation**: Douglas-Peucker with epsilon = 2% of perimeter

**Strengths:**
- Excellent for high-contrast edges
- Works well with good lighting
- Detects cards with clear boundaries

**Weaknesses:**
- Sensitive to noise and texture
- May miss cards in poor lighting
- Can detect false edges from background

**Debug Output:**
- `04_canny_20_60.png` - Low threshold (most sensitive)
- `04_canny_50_150.png` - Medium threshold
- `04_canny_100_250.png` - High threshold (least noise)
- `07_canny_morphology.png` - Morphologically closed edges (best threshold)
- `08_canny_contours.png` - All quadrilateral candidates overlaid (cyan)

---

### Strategy 2: Adaptive Thresholding

**Purpose:** Handle varying lighting conditions across the image.

**Approach:**
- Uses local adaptive thresholds instead of global
- Tries both normal and inverted thresholds
- Tests multiple block sizes for different scales

**Algorithm:**
```python
for (block_size, C) in [(11,2), (21,5), (31,10), (51,10)]:
    # 1. Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        filtered, 255,
        ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY,
        block_size, C
    )

    # 2. Try both normal and inverted
    for img in [thresh, 255 - thresh]:
        # 3. Find contours
        contours = cv2.findContours(img, RETR_LIST, CHAIN_APPROX_SIMPLE)

        # 4. Extract quadrilaterals
        quads = approximate_to_quads(contours)
        candidates.extend(quads)
```

**Parameters:**
- **Block Sizes**: 11×11, 21×21, 31×31, 51×51 (local neighborhood)
- **C Constant**: 2, 5, 10 (threshold adjustment)
- **Method**: Gaussian weighted mean of neighborhood

**Strengths:**
- Robust to uneven lighting
- Works with shadows or gradients
- Good for outdoor/natural lighting

**Weaknesses:**
- Computationally intensive
- May over-segment textured surfaces
- Block size affects detection scale

**Debug Output:**
- `09_adaptive_11_2.png` - Small block (11×11, sensitive to local features)
- `10_adaptive_31_10.png` - Large block (31×31, smoother regions)
- `11_adaptive_contours.png` - All quadrilateral candidates overlaid (orange)

---

### Strategy 3: Otsu's Thresholding

**Purpose:** Automatically find optimal global threshold.

**Approach:**
- Uses Otsu's method to compute optimal threshold
- Tries both binary and inverted binary
- Applies morphological smoothing

**Algorithm:**
```python
# 1. Compute optimal threshold automatically
_, otsu = cv2.threshold(
    filtered, 0, 255,
    THRESH_BINARY + THRESH_OTSU
)

# 2. Invert
otsu_inverted = 255 - otsu

# 3. Process both versions
for img in [otsu, otsu_inverted]:
    # 4. Morphological closing to smooth
    kernel = 3×3 rectangle
    img_morphed = cv2.morphologyEx(img, MORPH_CLOSE, kernel)

    # 5. Find contours
    contours = cv2.findContours(img_morphed, RETR_LIST, CHAIN_APPROX_SIMPLE)

    # 6. Extract quadrilaterals
    quads = approximate_to_quads(contours)
    candidates.extend(quads)
```

**Parameters:**
- **Threshold**: Automatically computed by Otsu's algorithm
- **Morphology Kernel**: 3×3 rectangle for smoothing
- **Method**: Minimizes intra-class variance

**Strengths:**
- No parameter tuning needed
- Fast computation
- Optimal for bimodal histograms

**Weaknesses:**
- Assumes bimodal distribution
- Single global threshold
- Poor with complex backgrounds

**Debug Output:**
- `12_otsu_binary.png` - Binary threshold result
- `13_otsu_inverted.png` - Inverted binary result
- `14_otsu_contours.png` - All quadrilateral candidates overlaid (magenta)

---

### Strategy 4: Color-Based Segmentation

**Purpose:** Detect gray/metallic credit cards by color properties.

**Approach:**
- Uses HSV color space to find low-saturation (gray) regions
- Combines saturation and value channels
- Applies morphological operations to clean regions

**Algorithm:**
```python
# 1. Convert to HSV color space
hsv = cv2.cvtColor(image, COLOR_BGR2HSV)
sat = hsv[:,:,1]  # Saturation channel
val = hsv[:,:,2]  # Value channel

# 2. Detect low saturation (gray colors)
_, low_sat_mask = cv2.threshold(sat, 30, 255, THRESH_BINARY_INV)
# sat < 30 → gray/metallic

# 3. Combine with value range (not too dark, not too bright)
gray_mask = cv2.bitwise_and(
    low_sat_mask,
    cv2.inRange(val, 80, 200)  # 80 ≤ brightness ≤ 200
)

# 4. Morphological cleanup
kernel = 7×7 rectangle
gray_mask = cv2.morphologyEx(gray_mask, MORPH_CLOSE, kernel)
gray_mask = cv2.morphologyEx(gray_mask, MORPH_OPEN, kernel)

# 5. Find contours
contours = cv2.findContours(gray_mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

# 6. Extract quadrilaterals (more aggressive approximation)
quads = approximate_to_quads(contours, epsilon_factor=0.03)
candidates.extend(quads)
```

**Parameters:**
- **Saturation Threshold**: < 30 (detects gray/metallic)
- **Value Range**: 80-200 (excludes very dark/bright)
- **Morphology Kernel**: 7×7 rectangle for cleanup
- **Approximation**: 3% of perimeter (more aggressive)

**Strengths:**
- Works well with metallic/gray cards
- Robust to lighting direction
- Good with colored backgrounds

**Weaknesses:**
- Fails with colored cards
- May confuse other gray objects
- Sensitive to white balance

**Debug Output:**
- `15_hsv_saturation.png` - Saturation channel visualization
- `16_low_sat_mask.png` - Low saturation mask (gray regions)
- `17_gray_mask.png` - Final gray mask after morphology
- `18_color_contours.png` - All quadrilateral candidates overlaid (green)

---

### Strategy Comparison

| Strategy | Best For | Computational Cost | Robustness | Parameters |
|----------|----------|-------------------|------------|------------|
| **Canny** | High contrast, clear edges | Medium | Medium | 5 threshold pairs |
| **Adaptive** | Uneven lighting, shadows | High | High | 4 block sizes |
| **Otsu** | Bimodal lighting, simple scenes | Low | Low | Auto (0) |
| **Color** | Metallic cards, colored backgrounds | Medium | Medium | 2 thresholds |

**Combined Approach:**
- All strategies run in parallel
- Results are pooled together
- Best candidate selected from combined pool
- Provides robustness across diverse conditions

---

## Candidate Scoring & Selection

After all strategies generate candidates, the system scores and selects the best one.

### Quadrilateral Extraction

Before scoring, raw contours are approximated to quadrilaterals:

```python
def approximate_to_quads(contours, epsilon_factor=0.02):
    quads = []
    for contour in contours:
        # Douglas-Peucker approximation
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Keep only 4-sided polygons
        if len(approx) == 4:
            quads.append(approx)

    return quads
```

**Parameters:**
- **Epsilon Factor**: 2% (default) or 3% (color strategy)
- Higher epsilon = more aggressive simplification

---

### Validation Filters (Hard Constraints)

Each candidate must pass **4 validation filters** or receives score = 0.0:

#### Filter 1: Area Ratio Check

```python
MIN_CARD_AREA_RATIO = 0.01  # 1% of image
MAX_CARD_AREA_RATIO = 0.5   # 50% of image

area = cv2.contourArea(corners)
area_ratio = area / image_area

if area_ratio < 0.01 or area_ratio > 0.5:
    reject("area_ratio too small/large")
```

**Rationale:**
- Cards too small (< 1%) are likely noise
- Cards too large (> 50%) are unrealistic
- Typical good ratio: 5-15% of image

**Rejection Example:** `"area_ratio=0.003"` → Too small

---

#### Filter 2: Aspect Ratio Check

```python
CARD_ASPECT_RATIO = 1.586  # ISO/IEC 7810 ID-1 standard
# 85.60 mm ÷ 53.98 mm = 1.5858...

aspect_ratio_tolerance = 0.15  # ±15%

width, height = get_quad_dimensions(corners)
aspect_ratio = max(width, height) / min(width, height)

ratio_diff = abs(aspect_ratio - 1.586) / 1.586
if ratio_diff > 0.15:
    reject("aspect_ratio mismatch")
```

**Rationale:**
- Credit cards have standardized dimensions
- Tolerance accounts for perspective distortion
- Accepted range: 1.35 - 1.82

**Rejection Example:** `"aspect_ratio=2.1, expected~1.586"` → Too elongated

---

#### Filter 3: Corner Angle Check

```python
CORNER_ANGLE_TOLERANCE = 25°  # Degrees from 90°

angles = compute_corner_angles(corners)  # [angle1, angle2, angle3, angle4]
angle_deviations = [abs(angle - 90) for angle in angles]
max_deviation = max(angle_deviations)

if max_deviation > 25:
    reject("corner_angle_deviation too large")
```

**Algorithm for Corner Angles:**
```python
def compute_corner_angles(corners):
    angles = []
    for i in range(4):
        p1 = corners[(i-1) % 4]  # Previous point
        p2 = corners[i]          # Current point
        p3 = corners[(i+1) % 4]  # Next point

        v1 = p1 - p2  # Vector from p2 to p1
        v2 = p3 - p2  # Vector from p2 to p3

        cos_angle = dot(v1, v2) / (norm(v1) * norm(v2))
        angle = arccos(clamp(cos_angle, -1, 1)) * 180/π
        angles.append(angle)

    return angles
```

**Rationale:**
- Credit cards are rectangular
- Perspective can distort but not excessively
- Accepted range per corner: 65° - 115°

**Rejection Example:** `"corner_angle_deviation=35°"` → Too skewed

---

#### Filter 4: Convexity Check

```python
if not cv2.isContourConvex(corners):
    reject("not_convex")
```

**Rationale:**
- Credit cards are always convex quadrilaterals
- Concave shapes indicate incorrect detection
- Simple and fast validation

**Rejection Example:** `"not_convex"` → Invalid shape

---

### Scoring Function (Soft Metrics)

Candidates passing all filters receive a score from 0.0 to 1.0 based on 3 weighted components:

#### Component 1: Area Score (Weight: 40%)

```python
area_score = min(area_ratio / 0.1, 1.0)
```

**Formula:**
- Peak score at 10% of image area
- Linear scaling from 0% to 10%
- Capped at 1.0 for larger areas

**Examples:**
- 5% of image → score = 0.5
- 10% of image → score = 1.0
- 20% of image → score = 1.0 (capped)

**Rationale:** Larger cards are more reliably detected and measured.

---

#### Component 2: Aspect Ratio Score (Weight: 30%)

```python
ratio_diff = abs(aspect_ratio - 1.586) / 1.586
ratio_score = 1.0 - ratio_diff / 0.15
```

**Formula:**
- Perfect match (1.586) → score = 1.0
- Maximum tolerance (15%) → score = 0.0
- Linear interpolation between

**Examples:**
- Aspect ratio 1.586 → score = 1.0 (perfect)
- Aspect ratio 1.65 → score = 0.73
- Aspect ratio 1.35 → score = 0.01 (barely passing)

**Rationale:** Closer to standard dimensions indicates correct detection.

---

#### Component 3: Angle Score (Weight: 30%)

```python
max_deviation = max(abs(angle - 90) for angle in corner_angles)
angle_score = 1.0 - max_deviation / 25
```

**Formula:**
- Perfect 90° corners → score = 1.0
- Maximum tolerance (25°) → score = 0.0
- Uses worst corner (maximum deviation)

**Examples:**
- Corners: [90°, 89°, 91°, 90°] → score = 1.0 (perfect)
- Corners: [85°, 92°, 88°, 95°] → score = 0.80 (max dev = 5°)
- Corners: [75°, 105°, 80°, 100°] → score = 0.40 (max dev = 15°)

**Rationale:** More rectangular shapes are more reliable card detections.

---

#### Final Score Calculation

```python
final_score = (
    0.4 × area_score +
    0.3 × ratio_score +
    0.3 × angle_score
)
```

**Score Breakdown Example:**
```json
{
  "score": 0.83,
  "score_components": {
    "area": 0.85,    // Card is 8.5% of image
    "ratio": 0.95,   // Aspect ratio 1.60 (very close to 1.586)
    "angle": 0.70    // Corners range 83°-97° (decent rectangularity)
  }
}
```

**Calculation:**
```
final_score = 0.4 × 0.85 + 0.3 × 0.95 + 0.3 × 0.70
            = 0.34 + 0.285 + 0.21
            = 0.835 ≈ 0.83
```

---

### Selection Process

After scoring, the system selects the best candidate:

```python
# 1. Score all candidates from all strategies
best_score = 0.0
best_result = None
all_scored = []

for contour in candidates:
    corners = contour.reshape(4, 2)
    score, details = score_card_candidate(corners, image_area)

    all_scored.append((corners, score, details))

    if score > best_score:
        best_score = score
        best_result = details

# 2. Sort by score (descending)
all_scored.sort(key=lambda x: x[1], reverse=True)

# 3. Extract top 5 for visualization
top_candidates = all_scored[:5]

# 4. Apply minimum threshold
MINIMUM_SCORE_THRESHOLD = 0.3

if best_score < 0.3:
    return None  # Detection failed
else:
    return best_result  # Detection succeeded
```

**Debug Output:**
- `19_all_candidates.png` - All candidates from all strategies (pink/purple)
- `20_scored_candidates.png` - Top 5 ranked candidates with scores
  - #1 (Green) - Highest score (selected)
  - #2 (Yellow) - Second best
  - #3 (Orange) - Third best
  - #4 (Magenta) - Fourth best
  - #5 (Pink) - Fifth best
- `21_final_detection.png` - Final selected card with details

---

### Minimum Score Threshold

**Threshold:** 0.3 (30%)

**Rationale:**
- Prevents false positives
- Ensures minimum quality standards
- Empirically determined from testing

**What happens if score < 0.3:**
```
Detection Result: Failed
Fail Reason: "card_not_detected"
Output: No card information, confidence = 0
```

**Common scenarios:**
- No card in image
- Card heavily occluded
- Extreme perspective distortion
- Poor image quality

---

### Complete Pipeline Summary

```
Input Image
    ↓
Preprocessing (Grayscale + Filter)
    ↓
┌────────────────────────────────────┐
│ Parallel Strategy Execution        │
├────────────────────────────────────┤
│ • Canny (5 configs) → ~20 quads    │
│ • Adaptive (4 configs) → ~15 quads │
│ • Otsu (2 versions) → ~10 quads    │
│ • Color (HSV filter) → ~8 quads    │
└────────────────────────────────────┘
    ↓
Candidate Pool (~50-100 quadrilaterals)
    ↓
Validation Filters (Hard Pass/Fail)
├─ Area: 1% - 50% of image
├─ Aspect Ratio: 1.35 - 1.82
├─ Corner Angles: 65° - 115°
└─ Convexity: Must be convex
    ↓
Valid Candidates (~5-20)
    ↓
Scoring (0.0 - 1.0)
├─ 40% Area Score
├─ 30% Ratio Score
└─ 30% Angle Score
    ↓
Sort by Score (Descending)
    ↓
Top 5 → 20_scored_candidates.png
    ↓
Best Score ≥ 0.3?
├─ Yes → 21_final_detection.png ✅
└─ No → Detection Failed ❌
```

---

**Related Documentation:**
- [03-scale-calibration.md](03-scale-calibration.md) - Using detected card for px/cm conversion
- [Algorithm Index](README.md) - All algorithm documentation

**Source Code:** `src/card_detection.py`

---

**Last Updated:** 2026-02-03
