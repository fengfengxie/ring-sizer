# PRD: Finger Outer Diameter Measurement from Single Image (v0)

## 1. Purpose

Build a **local, terminal-executable computer vision program** that measures the **outer width (diameter) of a finger at the ring-wearing zone**, using a **single RGB image** containing:
- one hand with an intended ring-wearing finger
- one **standard credit card** used as a physical size reference

This measurement (in cm) serves as the **core primitive** for all downstream ring size estimation logic.

---

## 2. Scope (v0)

### In Scope
- Single image input
- Credit card–based scale calibration
- Finger contour extraction
- Ring-wearing zone localization
- Finger outer diameter measurement (cm)
- Debug visualization output
- Confidence / failure reporting

### Out of Scope
- Ring size classification (US/EU sizes)
- Knuckle modeling
- Finger swelling over time
- Depth / LiDAR usage
- Web or mobile UI
- ML training (pretrained models allowed)

---

## 3. Input Specification

### 3.1 Input Image Requirements
- RGB image (JPG / PNG)
- Resolution ≥ 1080p recommended
- Captured from near top-down view
- Finger and credit card lie on the **same plane**
- Credit card must be fully visible (≥ 3 corners)

### 3.2 Reference Object
- Standard credit card
- Size: **85.60 mm × 53.98 mm** (ISO/IEC 7810 ID-1)

---

## 4. Output Specification

### 4.1 Numerical Outputs (JSON)
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
````

### 4.2 Visualization Output (PNG)

A debug image overlay showing:

* Credit card contour + corners
* Finger contour
* Ring-wearing zone band
* Sampled cross-sections
* Final measured width annotation

---

## 5. Definitions (Important)

### 5.1 Finger Outer Diameter

The **external horizontal width** of the finger (soft tissue included) measured at the **ring-wearing zone**.

> This is **NOT** the inner diameter of a ring.
> It is a geometric proxy used for later mapping.

### 5.2 Ring-Wearing Zone (v0 Definition)

A fixed region along the finger axis:

* Located near the finger base (close to palm)
* Defined as **15%–25% of the finger length from the palm-side end**
* Width measured by sampling multiple cross-sections within this band

---

## 6. Program Interface (CLI)

### 6.1 Command

```bash
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --debug debug_overlay.png
```

### 6.2 Optional Flags

```bash
--save-intermediate        # save masks, contours, rectified card
--finger-index auto|index|middle|ring|pinky
--confidence-threshold 0.7
```

---

## 7. Processing Pipeline

### 7.1 Step 1: Image Quality Check

* Blur detection (e.g., Laplacian variance)
* Exposure / contrast check
* Early exit if quality too low

### 7.2 Step 2: Credit Card Detection & Scale Calibration

* Detect card contour (quadrilateral)
* Verify aspect ratio (~1.586)
* Perspective rectify card region
* Compute `px_per_cm`

Fail if:

* Card not found
* Aspect ratio deviates beyond threshold

### 7.3 Step 3: Hand & Finger Segmentation

* Use hand segmentation or landmark model (pretrained allowed)
* Extract hand mask
* Identify finger region (largest extended finger by default)
* Clean mask (morphology, largest component)

### 7.4 Step 4: Finger Contour Extraction

* Extract outer contour of finger mask
* Smooth contour (optional spline / Gaussian)

### 7.5 Step 5: Finger Axis Estimation

* Compute principal axis (PCA or skeleton)
* Define longitudinal direction

### 7.6 Step 6: Ring-Wearing Zone Localization

* Project finger mask onto axis
* Identify palm-side end
* Define zone at 15%–25% of finger length

### 7.7 Step 7: Width Measurement

* Sample N (e.g., 20) cross-sections perpendicular to axis
* Compute left/right intersections with contour
* Convert widths from px → cm
* Use median or trimmed mean

### 7.8 Step 8: Confidence Scoring

Confidence influenced by:

* Card detection reliability
* Finger mask stability
* Width variance across samples
* Viewing angle proxy (card distortion)

---

## 8. Failure Modes & Handling

| Condition             | Action            |
| --------------------- | ----------------- |
| Card not detected     | Fail with reason  |
| Card heavily tilted   | Low confidence    |
| Finger mask too small | Fail              |
| High width variance   | Reduce confidence |
| Image blur            | Fail or warn      |

---

## 9. Confidence Definition

```text
confidence ∈ [0, 1]
```

Suggested mapping:

* > 0.85: High confidence
* 0.6–0.85: Medium
* <0.6: Low (recommend re-capture)

---

## 10. Validation Strategy

### 10.1 Visual Validation

* Human inspection of debug overlays

### 10.2 Numerical Sanity Checks

* Finger width in realistic range (1.4–2.4 cm)
* Cross-section variance < threshold

---

## 11. Directory Structure (Suggested)

```text
project/
├── measure_finger.py
├── models/
├── utils/
│   ├── card_detection.py
│   ├── finger_segmentation.py
│   ├── geometry.py
├── samples/
├── outputs/
└── README.md
```

---

## 12. Definition of Done (v0)

* Program runs locally from terminal
* Produces numeric output + debug image
* Measurement stable across repeated runs
* Clear failure reasons when applicable
