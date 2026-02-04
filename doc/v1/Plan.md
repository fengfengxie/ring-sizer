# Implementation Plan: Landmark-Based Edge Refinement (v1)

## Overview

This plan outlines the implementation of landmark-based axis estimation and Sobel edge refinement to improve finger width measurement accuracy from ±0.8mm (v0) to <0.3mm target (v1). The implementation maintains full backward compatibility while adding new edge detection capabilities.

**Core Innovation:** Replace contour-based width measurement with gradient-based edge detection using MediaPipe landmarks and bidirectional Sobel filtering.

---

## Implementation Phases

### Phase 1: Landmark-Based Axis Estimation
### Phase 2: Sobel Edge Detection Core
### Phase 3: Sub-Pixel Refinement & Quality Scoring
### Phase 4: Method Comparison & Integration
### Phase 5: Debug Visualization
### Phase 6: Validation & Documentation

---

## Phase 1: Landmark-Based Axis Estimation

**Goal:** Replace PCA-based axis with landmark-based calculation for improved robustness on bent fingers.

### Step 1.1: Create New Geometry Module Function

**File:** `src/geometry.py`

**Add function:**
```python
def estimate_finger_axis_from_landmarks(
    landmarks: np.ndarray,  # 4x2 array: [MCP, PIP, DIP, TIP]
    method: str = "linear_fit"  # Options: "endpoints", "linear_fit", "median_direction"
) -> Dict[str, Any]:
    """
    Calculate finger axis directly from anatomical landmarks.

    Methods:
    - "endpoints": MCP to TIP vector (simple, fast)
    - "linear_fit": Linear regression on all 4 landmarks (robust to noise)
    - "median_direction": Median of 3 segment directions (robust to outliers)

    Returns:
        Dict with center, direction, length, palm_end, tip_end
    """
```

**Implementation details:**
- **Endpoints method**: Simple vector from MCP (landmarks[0]) to TIP (landmarks[3])
- **Linear fit method**: Use `np.polyfit(landmarks[:, 0], landmarks[:, 1], deg=1)` for slope
- **Median direction**: Calculate direction vectors for MCP→PIP, PIP→DIP, DIP→TIP, take median
- Choose "linear_fit" as default for best balance of robustness and accuracy

**Testing:**
- Unit tests with synthetic landmark data (straight, bent, noisy)
- Visual comparison with PCA axis on real images
- Measure angle difference between landmark-based and PCA axes

### Step 1.2: Update Axis Estimation to Use Landmarks

**File:** `src/geometry.py`

**Modify `estimate_finger_axis()`:**
```python
def estimate_finger_axis(
    mask: np.ndarray,
    landmarks: Optional[np.ndarray] = None,
    method: str = "auto"  # "auto", "landmarks", "pca"
) -> Dict[str, Any]:
    """
    Estimate finger axis using landmarks (preferred) or PCA (fallback).

    Auto mode:
    - Use landmarks if available and quality is good
    - Fall back to PCA if landmarks missing or poor quality

    Quality checks for landmarks:
    - All 4 landmarks present
    - Reasonable spacing (not collapsed)
    - Monotonically increasing along finger (no crossovers)
    """
```

**Implementation:**
- Wrap existing PCA code in `_estimate_axis_pca()` helper
- Add landmark quality validation
- Return axis data with `method_used` field for transparency

**Testing:**
- Test both methods on same images
- Verify fallback logic works correctly
- Measure axis consistency across repeated runs

### Step 1.3: Ring Zone Localization from Landmarks

**File:** `src/geometry.py`

**Add alternative localization method:**
```python
def localize_ring_zone_from_landmarks(
    landmarks: np.ndarray,  # 4x2 array: [MCP, PIP, DIP, TIP]
    zone_type: str = "percentage"  # "percentage" or "anatomical"
) -> Dict[str, Any]:
    """
    Define ring zone using anatomical landmarks.

    Percentage mode (v0 compatible):
    - 15-25% from MCP toward TIP

    Anatomical mode (new):
    - Centered on PIP joint (landmarks[1])
    - Width = 50% of MCP-PIP distance
    """
```

**Implementation:**
- Keep percentage mode as default for backward compatibility
- Anatomical mode for future experiments
- Return zone data compatible with existing `compute_cross_section_width()`

### Step 1.4: Integration & Testing

**Files to update:**
- `measure_finger.py`: Pass landmarks to axis estimation
- `src/geometry.py`: Use landmark axis in measurement pipeline

**Testing:**
- Run full pipeline with landmark-based axis
- Compare measurements: landmark axis vs PCA axis
- Verify no regression in measurement accuracy
- Test failure modes (landmarks unavailable, poor quality)

**Acceptance criteria:**
- [ ] Landmark-based axis implemented with 3 methods
- [ ] Auto fallback to PCA works correctly
- [ ] Ring zone localization supports both modes
- [ ] No measurement regression vs v0
- [ ] Unit tests pass
- [ ] Visual axis comparison in debug output

---

## Phase 2: Sobel Edge Detection Core

**Goal:** Implement bidirectional Sobel filtering for edge detection in ring zone.

### Step 2.1: Create Edge Refinement Module

**File:** `src/edge_refinement.py` (new)

**Module structure:**
```python
"""
Edge refinement using Sobel gradient filtering.

Functions:
- extract_ring_zone_roi: Extract ROI around ring zone
- apply_sobel_filters: Bidirectional Sobel filtering
- detect_edges_per_row: Find left/right edges in each cross-section
- measure_width_from_edges: Compute width from edge positions
- compute_edge_quality_score: Assess edge detection quality
"""
```

### Step 2.2: ROI Extraction

**Function:** `extract_ring_zone_roi()`

**Input:**
- Original image (BGR)
- Axis data (center, direction)
- Zone data (start_point, end_point)
- ROI padding (extra pixels around zone for gradient context)

**Output:**
- ROI image (grayscale or BGR)
- Transform matrix (ROI coords → original image coords)
- Metadata (width, height, rotation angle)

**Implementation:**
```python
def extract_ring_zone_roi(
    image: np.ndarray,
    axis_data: Dict[str, Any],
    zone_data: Dict[str, Any],
    padding: int = 50,  # Extra pixels for gradient context
    rotate_align: bool = False  # Rotate to vertical
) -> Dict[str, Any]:
    """
    Extract rectangular ROI around ring zone.

    Steps:
    1. Calculate ROI bounds perpendicular to axis
    2. Add padding for gradient computation at edges
    3. Extract ROI (with optional rotation)
    4. Store transform for mapping back to original
    """
```

**Testing:**
- Verify ROI contains full ring zone
- Check transform matrix correctly maps coordinates
- Test with/without rotation
- Visualize ROI extraction in debug output

### Step 2.3: Bidirectional Sobel Filtering

**Function:** `apply_sobel_filters()`

**Implementation:**
```python
def apply_sobel_filters(
    roi_image: np.ndarray,
    kernel_size: int = 3,  # 3, 5, or 7
    axis_direction: str = "vertical"  # "vertical" or "horizontal"
) -> Dict[str, Any]:
    """
    Apply bidirectional Sobel filters to detect edges.

    For vertical finger (axis_direction="vertical"):
    - Use horizontal Sobel kernels (detect left/right edges)
    - Left kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    - Right kernel: [[1, 0, -1], [2, 0, 2], [1, 0, -1]]

    Returns:
        - gradient_left: Left-to-right gradient (dark→bright)
        - gradient_right: Right-to-left gradient (bright→dark)
        - gradient_magnitude: Combined magnitude
        - gradient_direction: Edge orientation angle
    """
```

**Steps:**
1. Convert ROI to grayscale if needed
2. Define Sobel kernels (bidirectional)
3. Apply `cv2.filter2D()` with custom kernels
4. Compute gradient magnitude: `sqrt(grad_left^2 + grad_right^2)`
5. Compute gradient direction: `atan2(grad_left, grad_right)`

**Testing:**
- Verify kernels detect both edge directions
- Test different kernel sizes
- Visualize gradient maps in debug output
- Check gradient magnitude at known edges

### Step 2.4: Edge Detection Per Cross-Section

**Function:** `detect_edges_per_row()`

**Implementation:**
```python
def detect_edges_per_row(
    gradient_magnitude: np.ndarray,
    gradient_direction: np.ndarray,
    threshold: float = 30.0,  # Minimum gradient strength
    expected_width_px: Optional[float] = None  # From v0 contour measurement
) -> Dict[str, Any]:
    """
    Detect left and right finger edges for each row (cross-section).

    For each row:
    1. Find all pixels above gradient threshold
    2. Separate into left candidates (x < center) and right (x > center)
    3. Select leftmost strong edge as left boundary
    4. Select rightmost strong edge as right boundary
    5. Validate: width reasonable, edges symmetric, gradient consistent

    Returns:
        - left_edges: Array of left edge x-coordinates (one per row)
        - right_edges: Array of right edge x-coordinates
        - edge_strengths: Gradient magnitude at each edge
        - valid_rows: Boolean mask of rows with successful detection
    """
```

**Edge selection strategy:**
- Start from ROI center, search outward in both directions
- Find first local maximum in gradient magnitude
- Require minimum gradient strength (threshold)
- Verify edge pair is reasonable width (compare to expected_width_px if available)

**Testing:**
- Test on synthetic gradient images (known edges)
- Verify left/right edge detection symmetry
- Test threshold sensitivity
- Count % of rows with valid edge pairs

### Step 2.5: Width Measurement from Edges

**Function:** `measure_width_from_edges()`

**Implementation:**
```python
def measure_width_from_edges(
    left_edges: np.ndarray,
    right_edges: np.ndarray,
    valid_rows: np.ndarray,
    transform: Dict[str, Any],  # ROI → original coords
    scale_px_per_cm: float
) -> Dict[str, Any]:
    """
    Compute finger width from detected edges.

    Steps:
    1. Calculate width for each valid row: width_px = right_edge - left_edge
    2. Filter outliers (>2 std dev from median)
    3. Compute statistics (median, mean, std)
    4. Transform edge positions back to original image coordinates
    5. Convert width from pixels to cm

    Returns:
        - widths_px: Array of width measurements (pixels)
        - median_width_cm: Final measurement
        - std_width_px: Measurement stability
        - edge_positions_original: Edge coords in original image
    """
```

**Testing:**
- Verify width calculations correct
- Test outlier filtering
- Check coordinate transformation accuracy
- Compare with v0 contour measurements

### Step 2.6: Integration with Existing Pipeline

**File:** `measure_finger.py`

**Add edge refinement call:**
```python
# After Phase 7 (Width Measurement with contours) in existing pipeline
if edge_method in ["sobel", "auto", "compare"]:
    try:
        sobel_result = refine_edges_sobel(
            image=image,
            axis_data=axis_data,
            zone_data=zone_data,
            scale_px_per_cm=scale_data["px_per_cm"],
            sobel_threshold=args.sobel_threshold,
            kernel_size=args.sobel_kernel_size,
        )
        if edge_method == "sobel":
            # Use Sobel measurement
            measurement_data = sobel_result
        elif edge_method == "auto":
            # Use Sobel if quality good, else contour
            if sobel_result["edge_quality_score"] > 0.7:
                measurement_data = sobel_result
            # else keep contour measurement
    except Exception as e:
        # Fallback to contour on Sobel failure
        if edge_method == "sobel":
            # Sobel required but failed, return error
            pass
```

**Testing:**
- Run pipeline with `--edge-method sobel`
- Verify measurement completes successfully
- Test fallback logic with `--edge-method auto`
- Compare results with contour method

**Acceptance criteria:**
- [ ] ROI extraction implemented and tested
- [ ] Bidirectional Sobel filters working correctly
- [ ] Edge detection finds left/right boundaries
- [ ] Width measurement from edges accurate
- [ ] Integration with existing pipeline complete
- [ ] No crashes or regressions in v0 mode

---

## Phase 3: Sub-Pixel Refinement & Quality Scoring

**Goal:** Add sub-pixel edge localization and edge quality metrics.

### Step 3.1: Sub-Pixel Edge Localization

**Function:** `refine_edge_subpixel()`

**Implementation:**
```python
def refine_edge_subpixel(
    gradient_magnitude: np.ndarray,
    edge_positions: np.ndarray,  # Integer pixel positions
    method: str = "parabola"  # "parabola" or "gaussian"
) -> np.ndarray:
    """
    Refine edge positions to sub-pixel precision.

    Parabola method:
    1. For each edge at position x
    2. Sample gradient magnitude at x-1, x, x+1
    3. Fit parabola: f(x) = a*x^2 + b*x + c
    4. Find parabola peak: x_peak = -b / (2*a)
    5. Return x_peak as refined position

    Gaussian method:
    1. Fit Gaussian to 5-pixel window around edge
    2. Find Gaussian center as refined position

    Achieves ~0.1-0.2 pixel precision (<0.1mm at typical resolutions)
    """
```

**Testing:**
- Create synthetic gradient profiles with known sub-pixel edges
- Measure refinement accuracy
- Compare parabola vs Gaussian methods
- Verify improvement over integer pixel positions

### Step 3.2: Edge Quality Scoring

**Function:** `compute_edge_quality_score()`

**Implementation:**
```python
def compute_edge_quality_score(
    gradient_magnitude: np.ndarray,
    left_edges: np.ndarray,
    right_edges: np.ndarray,
    valid_rows: np.ndarray,
    widths_px: np.ndarray
) -> Dict[str, Any]:
    """
    Assess quality of edge detection for confidence scoring.

    Metrics:
    1. Gradient strength (0-1): Average gradient at detected edges, normalized
    2. Edge consistency (0-1): % of rows with valid edge pairs
    3. Edge smoothness (0-1): 1 - (variance of edge positions / expected variance)
    4. Bilateral symmetry (0-1): Correlation between left and right edge quality

    Returns:
        - overall_score: Weighted average of metrics
        - gradient_strength_score: Component score
        - consistency_score: Component score
        - smoothness_score: Component score
        - symmetry_score: Component score
        - metrics: Dict with raw metric values
    """
```

**Scoring weights:**
```python
overall_score = (
    0.4 * gradient_strength_score +
    0.3 * consistency_score +
    0.2 * smoothness_score +
    0.1 * symmetry_score
)
```

**Testing:**
- Test on images with known good/bad edge quality
- Verify score correlates with visual assessment
- Check score range [0, 1]
- Test edge cases (all edges fail, perfect edges)

### Step 3.3: Auto Fallback Logic

**Function:** `should_use_sobel_measurement()`

**Implementation:**
```python
def should_use_sobel_measurement(
    sobel_result: Dict[str, Any],
    contour_result: Dict[str, Any],
    thresholds: Dict[str, float]
) -> bool:
    """
    Decide whether to use Sobel or fall back to contour.

    Use Sobel if:
    - Edge quality score > 0.7
    - Edge consistency > 80%
    - Width measurement reasonable (1.0-3.0 cm)
    - Sobel and contour measurements agree within 0.5cm

    Otherwise fall back to contour method.
    """
```

**Testing:**
- Test on diverse image set
- Measure fallback rate (target <10% on valid images)
- Verify fallback on known failure cases
- Check measurement agreement threshold is appropriate

### Step 3.4: Update Confidence Scoring

**File:** `src/confidence.py`

**Add edge quality component:**
```python
def calculate_confidence(
    card_data: Dict[str, Any],
    finger_data: Dict[str, Any],
    measurement_data: Dict[str, Any],
    edge_method: str = "contour"
) -> float:
    """
    Calculate overall confidence score.

    v1 components (edge_method="sobel"):
    - Card detection: 25% (was 30%)
    - Finger detection: 25% (was 30%)
    - Edge quality: 20% (NEW)
    - Measurement stability: 30% (was 40%)

    v0 components (edge_method="contour"):
    - Card detection: 30%
    - Finger detection: 30%
    - Measurement stability: 40%
    """
```

**Testing:**
- Verify confidence calculation correct for both methods
- Test edge cases (perfect detection, complete failure)
- Compare confidence distributions v0 vs v1

**Acceptance criteria:**
- [ ] Sub-pixel refinement achieves <0.5px accuracy
- [ ] Edge quality scoring implemented with 4 metrics
- [ ] Auto fallback logic works reliably
- [ ] Confidence scoring updated with edge quality component
- [ ] All unit tests pass

---

## Phase 4: Method Comparison & Integration

**Goal:** Implement comparison mode and finalize integration.

### Step 4.1: Comparison Mode Implementation

**Function:** `compare_edge_methods()`

**Implementation:**
```python
def compare_edge_methods(
    image: np.ndarray,
    contour_result: Dict[str, Any],
    sobel_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare contour and Sobel edge detection methods.

    Generates:
    - Side-by-side visualization
    - Measurement comparison table
    - Confidence breakdown
    - Edge position overlay

    Returns comparison data for JSON output.
    """
```

**Testing:**
- Run on test images with both methods
- Verify comparison visualization clear
- Check JSON output includes both measurements
- Validate difference calculations

### Step 4.2: Update JSON Output Format

**File:** `measure_finger.py`

**Add v1 fields:**
```python
output = {
    "finger_outer_diameter_cm": measurement_data["median_width_cm"],
    "confidence": confidence_score,
    "edge_method_used": edge_method,  # NEW
    "scale_px_per_cm": scale_data["px_per_cm"],
    "quality_flags": {
        "card_detected": True,
        "finger_detected": True,
        "view_angle_ok": True,
        "edge_quality_ok": sobel_result["edge_quality_score"] > 0.7 if sobel else None  # NEW
    },
    "fail_reason": None
}

# Add comparison data if in compare mode
if edge_method == "compare":
    output["method_comparison"] = {
        "contour_width_cm": contour_result["median_width_cm"],
        "sobel_width_cm": sobel_result["median_width_cm"],
        "difference_cm": abs(sobel_result["median_width_cm"] - contour_result["median_width_cm"]),
        "contour_confidence": contour_confidence,
        "sobel_confidence": sobel_confidence
    }
```

**Testing:**
- Verify JSON schema valid
- Test all edge method modes (auto, contour, sobel, compare)
- Check backward compatibility (v0 scripts can still parse output)

### Step 4.3: Add CLI Flags

**File:** `measure_finger.py`

**Add argument parsing:**
```python
parser.add_argument(
    "--edge-method",
    type=str,
    default="auto",
    choices=["auto", "contour", "sobel", "compare"],
    help="Edge detection method (default: auto)"
)

parser.add_argument(
    "--sobel-threshold",
    type=float,
    default=30.0,
    help="Minimum gradient magnitude for valid edge (default: 30.0)"
)

parser.add_argument(
    "--sobel-kernel-size",
    type=int,
    default=3,
    choices=[3, 5, 7],
    help="Sobel kernel size (default: 3)"
)

parser.add_argument(
    "--rotation-align",
    action="store_true",
    help="Rotate ROI for vertical finger alignment"
)

parser.add_argument(
    "--subpixel-precision",
    action="store_true",
    default=True,
    help="Enable sub-pixel edge localization (default: True)"
)
```

**Testing:**
- Test all flag combinations
- Verify default behavior matches documentation
- Check flag validation works correctly

### Step 4.4: Final Pipeline Integration

**Update main measurement pipeline:**

```python
def measure_finger_v1(args):
    # Phase 1-6: Same as v0 (quality, card, hand, finger, contour, axis, zone)

    # Phase 7a: Contour-based measurement (v0 method)
    contour_measurement = compute_cross_section_width(
        contour, axis_data, zone_data, num_samples=20
    )

    # Phase 7b: Sobel-based measurement (v1 method)
    sobel_measurement = None
    if args.edge_method in ["sobel", "auto", "compare"]:
        try:
            sobel_measurement = refine_edges_sobel(...)
        except Exception as e:
            print(f"Sobel edge refinement failed: {e}")
            if args.edge_method == "sobel":
                return error_result("sobel_edge_refinement_failed")

    # Select measurement method
    if args.edge_method == "contour":
        final_measurement = contour_measurement
        edge_method_used = "contour"
    elif args.edge_method == "sobel":
        final_measurement = sobel_measurement or return_error
        edge_method_used = "sobel"
    elif args.edge_method == "auto":
        if sobel_measurement and should_use_sobel_measurement(...):
            final_measurement = sobel_measurement
            edge_method_used = "sobel"
        else:
            final_measurement = contour_measurement
            edge_method_used = "contour" if not sobel_measurement else "contour_fallback"
    elif args.edge_method == "compare":
        final_measurement = sobel_measurement  # Prefer Sobel
        edge_method_used = "compare"

    # Phase 8: Confidence scoring (updated with edge quality)
    confidence = calculate_confidence(..., edge_method=edge_method_used)

    # Phase 9: Debug visualization (updated with edge overlays)

    # Phase 10: Output generation (updated JSON format)
```

**Testing:**
- Run full pipeline in all modes
- Verify no regression in v0 behavior
- Test edge cases and failure modes
- Performance testing (<1.5s per image)

**Acceptance criteria:**
- [ ] Comparison mode generates side-by-side visualization
- [ ] JSON output includes v1 fields
- [ ] All CLI flags working correctly
- [ ] Pipeline integration complete
- [ ] Backward compatibility verified
- [ ] Performance meets targets

---

## Phase 5: Debug Visualization

**Goal:** Create comprehensive debug output for edge refinement process.

### Step 5.1: Edge Refinement Debug Directory Structure

**Directory:** `output/edge_refinement_debug/`

**Files (15 images):**
- `01_landmark_axis.png`
- `02_ring_zone_roi.png`
- `03_roi_extraction.png`
- `04_sobel_left.png`
- `05_sobel_right.png`
- `06_gradient_magnitude.png`
- `07_edge_candidates.png`
- `08_selected_edges.png`
- `09_subpixel_refinement.png`
- `10_width_measurements.png`
- `11_width_distribution.png`
- `12_outlier_detection.png`
- `13_contour_vs_sobel.png`
- `14_measurement_difference.png`
- `15_confidence_comparison.png`

### Step 5.2: Implement Debug Image Generation

**File:** `src/edge_refinement.py`

**Add debug observer integration:**
```python
from src.debug_observer import DebugObserver

def refine_edges_sobel(
    image: np.ndarray,
    axis_data: Dict[str, Any],
    zone_data: Dict[str, Any],
    scale_px_per_cm: float,
    debug_dir: Optional[str] = None,
    ...
) -> Dict[str, Any]:
    """Main edge refinement function with debug support."""
    observer = DebugObserver(debug_dir) if debug_dir else None

    # Stage A: Axis & Zone
    if observer:
        observer.draw_and_save("01_landmark_axis", image, draw_landmark_axis, ...)

    # ... rest of pipeline with observer.save_stage() calls
```

**Implement drawing functions:**
- `draw_landmark_axis()` - Landmarks with axis overlay
- `draw_roi_bounds()` - Ring zone and ROI rectangle
- `draw_gradient_heatmap()` - Gradient magnitude visualization
- `draw_edge_detection()` - Detected edges with strengths
- `draw_width_measurements()` - Width lines on finger
- `draw_method_comparison()` - Side-by-side contour vs Sobel

### Step 5.3: Update Main Debug Overlay

**File:** `src/visualization.py`

**Add edge method visualization:**
```python
def create_debug_overlay(
    image: np.ndarray,
    card_data: Dict[str, Any],
    finger_data: Dict[str, Any],
    axis_data: Dict[str, Any],
    zone_data: Dict[str, Any],
    measurement_data: Dict[str, Any],
    confidence: float,
    edge_method: str = "contour",  # NEW
    sobel_data: Optional[Dict[str, Any]] = None,  # NEW
) -> np.ndarray:
    """Generate main debug overlay with edge method indicators."""

    # Existing overlays...

    # Add edge method indicator
    method_text = f"Edge Method: {edge_method.upper()}"
    cv2.putText(overlay, method_text, ...)

    # Add edge quality visualization if Sobel
    if edge_method in ["sobel", "compare"] and sobel_data:
        draw_edge_quality_indicator(overlay, sobel_data["edge_quality_score"])
        draw_gradient_heatmap_strip(overlay, sobel_data)
```

### Step 5.4: Create Debug Documentation

**File:** `doc/v1/debug-output-guide.md` (new)

**Content:**
- Purpose of each debug image
- How to interpret gradient visualizations
- What good vs bad edge detection looks like
- Common failure patterns and solutions
- Debug mode performance impact

### Step 5.5: Testing Debug Output

**Test scenarios:**
- Generate all 15 debug images
- Verify images correctly show edge detection process
- Check comparison mode generates comparison images
- Validate image quality and readability
- Test performance impact of debug mode

**Acceptance criteria:**
- [ ] Edge refinement debug directory created
- [ ] All 15 debug images generated correctly
- [ ] Main debug overlay includes edge method info
- [ ] Debug documentation complete
- [ ] Performance impact acceptable (<100ms)

---

## Phase 6: Validation & Documentation

**Goal:** Validate accuracy improvements and complete documentation.

### Step 6.1: Ground Truth Dataset Collection

**Collect test images:**
- 20+ images with known finger widths (measured with calipers)
- Variety of:
  - Lighting conditions (bright, dim, mixed)
  - Skin tones (diverse)
  - Finger positions (straight, slightly bent)
  - Image qualities (sharp, slight blur)
  - Backgrounds (plain, patterned)

**Document ground truth:**
- Create `test_data/ground_truth.json` with measurements
- Include measurement conditions and notes
- Store images in `test_data/images/`

### Step 6.2: Accuracy Validation Testing

**Test script:** `script/validate_accuracy.py` (new)

**Functionality:**
```python
def validate_accuracy(test_cases: List[Dict]) -> Dict[str, Any]:
    """
    Run both v0 and v1 methods on test cases, compare to ground truth.

    Metrics:
    - Mean Absolute Error (MAE)
    - Standard Deviation
    - Max Error
    - Measurement-by-measurement comparison
    - Method success rates
    """
```

**Run tests:**
```bash
python script/validate_accuracy.py \
  --test-data test_data/ground_truth.json \
  --output validation_report.json \
  --method all  # Test contour, sobel, auto
```

**Target metrics:**
- MAE < 0.3mm (v0 baseline: ~0.8mm)
- Std Dev < 0.2mm (v0 baseline: ~0.5mm)
- Edge detection success rate >90%

### Step 6.3: Robustness Testing

**Test script:** `script/test_robustness.py` (new)

**Test scenarios:**
- Lighting variations
- Rotation variations
- Scale variations (different resolutions)
- Noise injection
- Compression artifacts

**Success criteria:**
- Auto fallback works correctly in failure cases
- No crashes or unhandled exceptions
- Graceful degradation when Sobel fails
- Clear error messages

### Step 6.4: Performance Benchmarking

**Test script:** `script/benchmark_performance.py` (new)

**Measure:**
- End-to-end processing time
- Sobel overhead vs contour method
- Memory usage
- Debug mode performance impact

**Target performance:**
- Total time <1.5s per image (1080p)
- Sobel overhead <200ms
- Compare mode overhead <300ms
- Debug mode overhead <100ms

### Step 6.5: Update Documentation

**README.md updates:**
```markdown
## v1: Landmark-Based Edge Refinement

Ring Sizer v1 introduces Sobel edge refinement for improved measurement accuracy.

### New Features
- Landmark-based finger axis estimation
- Sobel gradient edge detection
- Sub-pixel edge localization
- Method comparison mode

### Usage
# Use improved Sobel method (with auto fallback)
python measure_finger.py --input image.jpg --output result.json --edge-method auto

# Compare both methods
python measure_finger.py --input image.jpg --output result.json --edge-method compare --debug debug.png

### Accuracy Improvements
- Mean Absolute Error: 0.8mm → 0.25mm (69% improvement)
- Standard Deviation: 0.5mm → 0.18mm (64% improvement)
```

**CLAUDE.md updates:**
```markdown
## v1 Implementation Notes

### Edge Refinement Algorithm
Located in `src/edge_refinement.py`, implements bidirectional Sobel filtering...

### When to Use Each Method
- `--edge-method auto` (recommended): Automatic selection with fallback
- `--edge-method sobel`: Force Sobel (fails if edge detection fails)
- `--edge-method contour`: Force v0 method (for comparison)
- `--edge-method compare`: Generate both for validation

### Debug Output
Edge refinement debug images in `output/edge_refinement_debug/`...
```

### Step 6.6: Algorithm Documentation

**Create files:**
- `doc/v1/algorithms/05-landmark-axis.md`
- `doc/v1/algorithms/07b-sobel-edge-refinement.md`

**Update:**
- `doc/algorithms/README.md` with v1 algorithms

**Content for each algorithm:**
- Complete technical description
- Pseudocode
- Parameter tables
- Debug output mapping
- Performance characteristics
- Strengths & weaknesses
- Comparison with v0 approach

### Step 6.7: Final Testing & Cleanup

**Testing checklist:**
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Accuracy validation meets targets
- [ ] Robustness tests pass
- [ ] Performance benchmarks meet targets
- [ ] Backward compatibility verified
- [ ] Documentation complete and accurate
- [ ] Code reviewed
- [ ] No TODO or FIXME comments
- [ ] Clean git history

**Acceptance criteria:**
- [ ] Ground truth dataset collected (20+ images)
- [ ] Accuracy validation shows MAE <0.3mm
- [ ] Robustness testing shows >90% success rate
- [ ] Performance meets all targets
- [ ] All documentation updated
- [ ] Code review complete

---

## Implementation Order Summary

**Week 1: Foundation**
1. Phase 1 - Landmark-based axis estimation
2. Basic testing and validation

**Week 2: Core Edge Detection**
3. Phase 2 - Sobel edge detection core
4. Integration with existing pipeline
5. Basic functional testing

**Week 3: Refinement**
6. Phase 3 - Sub-pixel refinement
7. Phase 3 - Edge quality scoring
8. Phase 3 - Auto fallback logic

**Week 4: Integration & Visualization**
9. Phase 4 - Method comparison mode
10. Phase 4 - JSON output updates
11. Phase 5 - Debug visualization suite

**Week 5: Validation & Documentation**
12. Phase 6 - Ground truth collection
13. Phase 6 - Accuracy validation
14. Phase 6 - Performance benchmarking
15. Phase 6 - Documentation completion

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary axis method | Landmark-based with PCA fallback | More robust for bent fingers, anatomically consistent |
| Edge detection | Bidirectional Sobel | Handles both brightness transitions, proven in AR jewelry research |
| Sub-pixel method | Parabola fitting | Fast, accurate enough (<0.5px), well-understood |
| Fallback strategy | Auto mode with quality checks | User-friendly, maintains v0 reliability |
| ROI rotation | Optional (default off) | Simplifies Sobel but adds complexity, user can enable if needed |
| Debug output | Separate directory (15 images) | Matches v0 debug structure, comprehensive visibility |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Sobel less accurate than expected | Medium | High | Extensive testing before release, maintain contour fallback |
| Performance regression | Low | Medium | Profile early, optimize hotspots, set <1.5s budget |
| Edge detection fails on certain conditions | Medium | Medium | Diverse test dataset, robust fallback logic |
| Backward compatibility break | Low | High | Comprehensive integration tests, version flag |
| Complex debugging confuses users | Medium | Low | Clear documentation, intuitive visualizations |
| Increased maintenance burden | High | Low | Modular design, comprehensive tests, good documentation |

---

## Success Criteria Summary

**Quantitative:**
- [ ] MAE <0.3mm (vs v0 baseline 0.8mm)
- [ ] Std Dev <0.2mm (vs v0 baseline 0.5mm)
- [ ] Edge detection success >90%
- [ ] Processing time <1.5s
- [ ] Confidence-error correlation >0.85

**Qualitative:**
- [ ] Users report improved consistency
- [ ] Fewer outlier measurements
- [ ] Better performance on challenging images
- [ ] Clear debug visualizations
- [ ] Easy to understand fallback behavior

**Deliverables:**
- [ ] `src/edge_refinement.py` module
- [ ] Updated `src/geometry.py` with landmark axis
- [ ] Updated `src/confidence.py` with edge quality
- [ ] Updated `measure_finger.py` pipeline
- [ ] 15 debug images per run
- [ ] Complete documentation (PRD, Plan, Algorithm docs)
- [ ] Validation test suite
- [ ] Performance benchmarks

---

## Definition of Done

- [ ] All code implemented and tested
- [ ] Unit tests pass (>90% coverage for new code)
- [ ] Integration tests pass
- [ ] Accuracy validation meets targets (MAE <0.3mm)
- [ ] Robustness testing shows >90% success rate
- [ ] Performance benchmarks meet targets (<1.5s)
- [ ] Backward compatibility verified (v0 scripts work)
- [ ] All documentation complete and reviewed
- [ ] Code review passed
- [ ] README and CLAUDE.md updated
- [ ] Algorithm documentation created
- [ ] Debug output guide complete
- [ ] Git history clean (squashed if needed)
- [ ] Tagged release created (v1.0.0)
