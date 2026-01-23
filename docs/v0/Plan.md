# Implementation Plan: Finger Outer Diameter Measurement

## Overview

This plan outlines the development of a local, terminal-executable computer vision program that measures the outer width (diameter) of a finger at the ring-wearing zone using a single RGB image with a credit card as a physical size reference.

---

## Phase 1: Project Setup & Infrastructure

### Step 1.1: Initialize Project Structure
- Create the directory structure as specified in the PRD
- Set up `measure_finger.py` as the main entry point
- Create `utils/` module with placeholder files:
  - `card_detection.py`
  - `finger_segmentation.py`
  - `geometry.py`
- Create `models/`, `samples/`, and `outputs/` directories

### Step 1.2: Set Up Dependencies
- Create `requirements.txt` with core dependencies:
  - `opencv-python` - image processing
  - `numpy` - numerical operations
  - `mediapipe` - hand landmark detection (pretrained model)
  - `scipy` - geometric calculations, interpolation
  - `argparse` - CLI argument parsing (stdlib)
- Create a virtual environment setup script or instructions in README

### Step 1.3: Implement CLI Interface
- Implement argument parsing in `measure_finger.py`:
  - `--input` (required): input image path
  - `--output` (required): output JSON path
  - `--debug` (optional): debug overlay output path
  - `--save-intermediate` (optional): save intermediate artifacts
  - `--finger-index` (optional): auto|index|middle|ring|pinky
  - `--confidence-threshold` (optional): default 0.7

---

## Phase 2: Image Quality Assessment

### Step 2.1: Implement Blur Detection
- Use Laplacian variance method to detect image blur
- Define threshold for acceptable sharpness
- Return blur score and pass/fail flag

### Step 2.2: Implement Exposure/Contrast Check
- Calculate histogram statistics
- Check for over/under exposure
- Verify sufficient contrast for edge detection

### Step 2.3: Create Quality Check Pipeline
- Combine blur and exposure checks
- Implement early exit with descriptive failure reasons
- Add quality metrics to output JSON

---

## Phase 3: Credit Card Detection & Scale Calibration

### Step 3.1: Implement Card Contour Detection
- Convert image to grayscale
- Apply edge detection (Canny)
- Find contours and filter for quadrilaterals
- Score candidates by:
  - Area (reasonable size relative to image)
  - Convexity
  - Corner angle validity (~90°)

### Step 3.2: Implement Aspect Ratio Verification
- Calculate detected quadrilateral aspect ratio
- Compare against standard credit card ratio (85.60/53.98 ≈ 1.586)
- Define acceptable deviation threshold (e.g., ±10%)
- Reject or flag if ratio deviates significantly

### Step 3.3: Implement Perspective Rectification
- Order detected corners consistently (top-left, top-right, bottom-right, bottom-left)
- Compute perspective transform matrix
- Rectify card region to canonical view
- Verify rectified dimensions match expected ratio

### Step 3.4: Calculate Scale Factor (px_per_cm)
- Use known credit card dimensions (8.56 cm × 5.398 cm)
- Calculate pixels-per-cm from rectified card dimensions
- Store scale factor for downstream measurements
- Estimate calibration confidence based on detection quality

---

## Phase 4: Hand & Finger Segmentation

### Step 4.1: Integrate MediaPipe Hand Detection
- Initialize MediaPipe Hands solution
- Process input image to detect hand landmarks
- Extract 21 hand landmarks if detected
- Handle multi-hand scenarios (select primary hand)

### Step 4.2: Generate Hand Mask
- Create binary mask from hand landmarks
- Use convex hull or landmark-based polygon
- Apply morphological operations for cleanup:
  - Erosion to remove noise
  - Dilation to fill gaps
  - Opening/closing for smoothing

### Step 4.3: Implement Finger Isolation
- Use landmark positions to identify individual fingers
- Map finger indices:
  - Index: landmarks 5-8
  - Middle: landmarks 9-12
  - Ring: landmarks 13-16
  - Pinky: landmarks 17-20
- Create isolated finger mask based on selected finger
- Support auto-detection (largest extended finger)

### Step 4.4: Clean Finger Mask
- Extract largest connected component
- Apply morphological smoothing
- Validate minimum mask size
- Store cleaned mask for contour extraction

---

## Phase 5: Finger Contour & Axis Estimation

### Step 5.1: Extract Finger Contour
- Find contours in cleaned finger mask
- Select largest contour
- Optional: Apply contour smoothing (Gaussian or spline interpolation)
- Store contour points for measurement

### Step 5.2: Estimate Finger Axis (PCA Method)
- Collect all points within finger mask
- Apply Principal Component Analysis (PCA)
- Extract primary axis (direction of maximum variance)
- Compute axis center point and direction vector

### Step 5.3: Alternative Axis Estimation (Skeleton Method)
- Compute morphological skeleton of finger mask
- Extract centerline points
- Fit line or polynomial to skeleton
- Use as backup or validation for PCA result

### Step 5.4: Determine Finger Orientation
- Identify palm-side end vs fingertip
- Use landmark positions if available
- Otherwise use mask geometry (wider end = palm side)
- Store orientation for zone localization

---

## Phase 6: Ring-Wearing Zone Localization

### Step 6.1: Project Finger onto Axis
- Project finger mask points onto principal axis
- Calculate finger length along axis
- Identify start (palm-side) and end (fingertip) points

### Step 6.2: Define Ring-Wearing Zone
- Calculate zone boundaries:
  - Start: 15% of finger length from palm-side end
  - End: 25% of finger length from palm-side end
- Convert percentage positions to pixel coordinates
- Store zone boundaries for measurement

### Step 6.3: Validate Zone
- Verify zone falls within finger mask
- Check zone has sufficient width for measurement
- Flag if zone appears invalid

---

## Phase 7: Width Measurement

### Step 7.1: Generate Cross-Section Sample Points
- Create N sample lines (default 20) within ring-wearing zone
- Lines perpendicular to finger axis
- Evenly spaced between zone start and end

### Step 7.2: Compute Cross-Section Widths
- For each sample line:
  - Find intersection points with finger contour
  - Calculate left and right edge positions
  - Compute width in pixels
- Store all width measurements

### Step 7.3: Convert to Physical Units
- Apply scale factor (px_per_cm) to pixel widths
- Convert to centimeters
- Validate against realistic range (1.4–2.4 cm)

### Step 7.4: Aggregate Final Measurement
- Calculate median width (robust to outliers)
- Also compute trimmed mean for comparison
- Calculate variance/std for confidence scoring
- Select final measurement value

---

## Phase 8: Confidence Scoring

### Step 8.1: Card Detection Confidence
- Score based on:
  - Aspect ratio deviation from ideal
  - Corner detection quality
  - Perspective distortion amount
- Weight: ~30% of total confidence

### Step 8.2: Finger Detection Confidence
- Score based on:
  - Hand landmark detection confidence (from MediaPipe)
  - Mask area and shape validity
  - Contour smoothness
- Weight: ~30% of total confidence

### Step 8.3: Measurement Stability Confidence
- Score based on:
  - Variance of cross-section widths
  - Outlier ratio in samples
  - Consistency between median and mean
- Weight: ~40% of total confidence

### Step 8.4: Aggregate Confidence Score
- Combine component scores with weights
- Apply calibration/scaling to [0, 1] range
- Classify: High (>0.85), Medium (0.6-0.85), Low (<0.6)

---

## Phase 9: Debug Visualization

### Step 9.1: Draw Credit Card Overlay
- Draw detected quadrilateral contour
- Mark corner points with labels
- Display scale factor annotation

### Step 9.2: Draw Finger Overlay
- Draw finger contour
- Show finger axis line
- Highlight ring-wearing zone band

### Step 9.3: Draw Measurement Details
- Draw sampled cross-section lines
- Show intersection points
- Annotate final measured width

### Step 9.4: Compose Final Debug Image
- Layer all overlays on original image
- Add legend/key
- Add measurement result text
- Save as PNG

---

## Phase 10: Output Generation & Error Handling

### Step 10.1: Generate JSON Output
- Structure output as specified in PRD:
  ```json
  {
    "finger_outer_diameter_cm": <float>,
    "confidence": <float>,
    "scale_px_per_cm": <float>,
    "quality_flags": {
      "card_detected": <bool>,
      "finger_detected": <bool>,
      "view_angle_ok": <bool>
    },
    "fail_reason": <string|null>
  }
  ```
- Write to specified output path

### Step 10.2: Implement Failure Handling
- Define failure conditions and messages:
  - "card_not_detected": Credit card not found in image
  - "card_aspect_ratio_invalid": Card shape doesn't match standard
  - "hand_not_detected": No hand found in image
  - "finger_mask_too_small": Finger region insufficient
  - "image_too_blurry": Image quality below threshold
  - "measurement_unstable": High variance in measurements
- Return appropriate fail_reason in JSON

### Step 10.3: Save Intermediate Outputs (Optional)
- When `--save-intermediate` flag set, save:
  - Preprocessed image
  - Card detection mask and corners
  - Hand/finger segmentation mask
  - Rectified card image
  - Contour visualization
- Save to `outputs/intermediate/` subdirectory

---

## Phase 11: Testing & Validation

### Step 11.1: Create Test Dataset
- Collect sample images with known measurements
- Include variety of:
  - Lighting conditions
  - Hand positions
  - Finger selections
  - Image qualities

### Step 11.2: Implement Sanity Checks
- Verify finger width in realistic range (1.4–2.4 cm)
- Check cross-section variance below threshold
- Validate scale factor reasonableness

### Step 11.3: Visual Validation
- Generate debug overlays for all test images
- Manual inspection for correct detection
- Document any systematic issues

### Step 11.4: Measurement Stability Test
- Run same image multiple times
- Verify consistent results
- Document any non-determinism

---

## Phase 12: Documentation & Cleanup

### Step 12.1: Write README.md
- Installation instructions
- Usage examples
- Input requirements
- Output format documentation
- Troubleshooting guide

### Step 12.2: Add Code Documentation
- Docstrings for all public functions
- Type hints throughout
- Inline comments for complex algorithms

### Step 12.3: Final Cleanup
- Remove dead code
- Verify all error messages are helpful
- Ensure graceful handling of edge cases

---

## Implementation Order (Recommended)

1. **Phase 1** - Project setup (foundation)
2. **Phase 3** - Credit card detection (critical path, enables scale)
3. **Phase 4** - Hand/finger segmentation (critical path)
4. **Phase 5** - Contour and axis (depends on Phase 4)
5. **Phase 6** - Zone localization (depends on Phase 5)
6. **Phase 7** - Width measurement (depends on Phases 3, 5, 6)
7. **Phase 2** - Image quality (can add after core pipeline works)
8. **Phase 8** - Confidence scoring (needs all measurements)
9. **Phase 9** - Debug visualization (helpful throughout)
10. **Phase 10** - Output generation (finalize format)
11. **Phase 11** - Testing (ongoing)
12. **Phase 12** - Documentation (final)

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Hand detection model | MediaPipe Hands | Pretrained, fast, accurate, MIT license |
| Edge detection | Canny | Well-understood, tunable parameters |
| Axis estimation | PCA primary | Robust, handles varying finger orientations |
| Width aggregation | Median | Robust to outliers from contour noise |
| Contour smoothing | Gaussian blur on mask | Simple, effective for noise reduction |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Card detection fails in varied lighting | Use adaptive thresholding, multiple detection strategies |
| MediaPipe doesn't detect hand | Fallback to skin color segmentation (reduced accuracy) |
| Finger axis estimation unreliable | Validate with skeleton method, use landmarks when available |
| High measurement variance | Increase sample count, apply outlier rejection |
| Scale calibration inaccurate | Verify aspect ratio, estimate calibration confidence |

---

## Definition of Done Checklist

- [ ] Program runs from terminal with specified CLI interface
- [ ] Produces valid JSON output matching specification
- [ ] Generates debug visualization PNG when requested
- [ ] Measurements are stable across repeated runs
- [ ] Clear failure reasons provided when applicable
- [ ] Confidence scores reflect measurement reliability
- [ ] Works with various valid input images
- [ ] Documentation complete
