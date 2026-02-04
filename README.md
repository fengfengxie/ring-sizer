# Ring Sizer 💍📏

> A computer vision application for measuring finger outer diameter to calculate the required ring size for Femometer Smart Ring users

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📖 Overview

Ring Sizer is a **local, terminal-executable computer vision program** that precisely measures the outer width (diameter) of a finger at the ring-wearing zone using a single RGB image. By leveraging a standard credit card as a physical size reference, the system provides accurate measurements essential for ring size estimation.

### Key Features

- 🎯 **Single Image Input**: No need for multiple photos or complex setups
- 💳 **Credit Card Calibration**: Uses standard credit card (ISO/IEC 7810 ID-1) for scale reference
- 🤖 **AI-Powered Detection**: MediaPipe-based hand and finger segmentation
- 🔬 **Dual Edge Detection** (v1): Landmark-based axis + Sobel gradient refinement for improved accuracy
- 📊 **Comprehensive Confidence Scoring**: Multi-factor quality assessment
- 🎨 **Debug Visualization**: Detailed overlay images for quality verification
- ⚡ **Fast & Local**: No cloud processing, runs entirely on your machine

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- A standard credit card (85.60 mm × 53.98 mm)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ring-sizer.git
cd ring-sizer

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
python measure_finger.py \
  --input input/test_image.jpg \
  --output output/result.json \
  --debug output/debug_overlay.png
```

### Quick Test

For quick testing without typing long commands:

```bash
# Auto-detect test image and generate debug output
./script/test.sh

# Test specific image
./script/test.sh input/my_image.jpg

# Run without debug visualization (faster)
./script/test.sh --no-debug
```

## 📋 Usage Examples

### Basic measurement (auto mode)
```bash
python measure_finger.py --input image.jpg --output result.json
```

### Generate debug visualization
```bash
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --debug debug_output.png
```

### Use Sobel edge refinement (v1)
```bash
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --edge-method sobel \
  --debug debug_output.png
```

### Compare edge detection methods
```bash
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --edge-method compare \
  --debug debug_output.png
```

### Specify finger and adjust Sobel parameters
```bash
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --finger-index ring \
  --edge-method sobel \
  --sobel-threshold 15.0 \
  --sobel-kernel-size 3
```

### Save intermediate processing artifacts
```bash
python measure_finger.py \
  --input image.jpg \
  --output result.json \
  --save-intermediate
```

## 📸 Input Requirements

For optimal results, ensure your input image meets these criteria:

- **Format**: JPG or PNG
- **Resolution**: 1080p or higher recommended
- **View angle**: Near top-down view
- **Contents**: 
  - One hand with the finger to be measured extended
  - One standard credit card fully visible (at least 3 corners)
  - Both finger and card on the same plane

## 📤 Output Format

### JSON Output (v0)
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
```

### JSON Output with v1 Edge Method Info
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

### JSON Output with Method Comparison
```json
{
  "finger_outer_diameter_cm": 1.78,
  "confidence": 0.87,
  "edge_method_used": "compare",
  "method_comparison": {
    "contour": {
      "width_cm": 1.82,
      "confidence": 0.86
    },
    "sobel": {
      "width_cm": 1.78,
      "edge_quality_score": 0.89,
      "confidence": 0.87
    },
    "difference": {
      "absolute_cm": -0.04,
      "relative_pct": -2.2
    },
    "recommendation": {
      "preferred_method": "sobel"
    }
  },
  "quality_flags": {
    "card_detected": true,
    "finger_detected": true,
    "view_angle_ok": true,
    "edge_quality_ok": true
  },
  "fail_reason": null
}
```

### Debug Visualization

When using the `--debug` flag, the program generates annotated images:

**Main Debug Overlay** (`output/debug.png`):
- ✅ Credit card contour and corners (green)
- 👆 Finger contour (magenta)
- 📏 Finger axis and endpoints (cyan/yellow)
- 🎯 Ring-wearing zone band (yellow, semi-transparent)
- 📊 Cross-section sampling lines (orange)
- 🔵 Measurement intersection points (blue)
- 📝 Final measurement and confidence annotation

**Additional Debug Directories** (when `--debug` used):
- `output/card_detection_debug/` - 21 images showing card detection pipeline
- `output/finger_segmentation_debug/` - 24 images showing finger isolation process
- `output/edge_refinement_debug/` - 12 images showing Sobel edge detection (v1)

## 🏗️ System Architecture

### Processing Pipeline (v1)

```
Input Image
    ↓
1. Image Quality Check
    ↓
2. Credit Card Detection & Scale Calibration
    ↓
3. Hand & Finger Segmentation (MediaPipe)
    ↓
4. Finger Contour Extraction
    ↓
5a. Finger Axis Estimation (Landmark-based) ← v1
5b. Fallback: PCA-based Axis (v0)
    ↓
6. Ring-Wearing Zone Localization (15-25% from palm)
    ↓
7a. Contour-Based Width Measurement (v0)
7b. Sobel Edge Refinement (v1) ← Optional
    ↓
8. Confidence Scoring (with Edge Quality v1)
    ↓
Output: JSON + Debug Visualization
```

### Key Components

| Module | Purpose |
|--------|---------|
| `card_detection.py` | Credit card detection, perspective correction, scale calibration |
| `finger_segmentation.py` | MediaPipe integration, hand/finger isolation, mask generation |
| `geometry.py` | Landmark/PCA axis estimation, zone localization, width measurement |
| `edge_refinement.py` | **[v1]** Sobel gradient edge detection, sub-pixel refinement |
| `image_quality.py` | Blur detection, exposure check, resolution validation |
| `confidence.py` | Multi-factor confidence scoring with edge quality (v1) |
| `visualization.py` | Debug overlay generation with detailed annotations |
| `debug_observer.py` | Debug pipeline infrastructure and drawing functions |

## 🔬 Technical Details

### Edge Detection Methods (v1)

The system supports two edge detection approaches:

**v0 Method: Contour-Based** (default for auto mode fallback)
- Extracts edges from morphologically processed finger mask
- Fast and reliable for well-segmented fingers
- Accuracy: ±0.5-2mm depending on mask quality

**v1 Method: Sobel Edge Refinement** (recommended)
- Landmark-based finger axis for improved accuracy
- Bidirectional Sobel gradient filtering
- Sub-pixel edge localization via parabola fitting
- Mask-constrained edge search (±10px around finger boundaries)
- Target accuracy: <0.5mm with sub-pixel precision
- 4-metric quality scoring (gradient strength, consistency, smoothness, symmetry)

**Auto Mode Behavior:**
- Attempts Sobel edge refinement first
- Falls back to contour method if edge quality insufficient (score <0.7)
- Transparent reporting of which method was used

### Ring-Wearing Zone Definition

The measurement is taken at the **ring-wearing zone**, defined as:
- Located **15%-25% of finger length** from the palm-side end
- Width measured by sampling **20 cross-sections** (contour) or **500+ cross-sections** (Sobel) within this zone
- Final measurement: **median width** across all samples (robust to outliers)

### Confidence Scoring

**v0 Confidence** (contour method):
| Component | Weight | Factors |
|-----------|--------|---------|
| **Card Detection** | 30% | Detection quality, aspect ratio, scale calibration |
| **Finger Detection** | 30% | Hand landmarks confidence, mask area validity |
| **Measurement Quality** | 40% | Variance, consistency, outlier ratio, range validation |

**v1 Confidence** (Sobel method):
| Component | Weight | Factors |
|-----------|--------|---------|
| **Card Detection** | 25% | Detection quality, aspect ratio, scale calibration |
| **Finger Detection** | 25% | Hand landmarks confidence, mask area validity |
| **Edge Quality** | 20% | Gradient strength, consistency, smoothness, symmetry |
| **Measurement Quality** | 30% | Variance, consistency, outlier ratio, range validation |

**Confidence Levels:**
- 🟢 **HIGH** (>0.85): Measurement is highly reliable
- 🟡 **MEDIUM** (0.6-0.85): Measurement is acceptable, minor issues detected
- 🔴 **LOW** (<0.6): Measurement may be unreliable, review recommended

## 📊 Performance Benchmarks

Based on testing with various image conditions:

| Metric | Value |
|--------|-------|
| Processing Time | ~1-3 seconds |
| Measurement Accuracy | ±0.1 cm (with high confidence) |
| Card Detection Rate | >95% (well-lit images) |
| Finger Detection Rate | >90% (full hand visible) |

## 🛠️ Development Status

### v0 (Baseline) ✅
✅ **Phase 1**: Project Setup & Infrastructure  
✅ **Phase 2**: Image Quality Assessment  
✅ **Phase 3**: Credit Card Detection & Scale Calibration  
✅ **Phase 4**: Hand & Finger Segmentation  
✅ **Phase 5**: Finger Contour & Axis Estimation  
✅ **Phase 6**: Ring-Wearing Zone Localization  
✅ **Phase 7**: Width Measurement  
✅ **Phase 8**: Confidence Scoring  
✅ **Phase 9**: Debug Visualization  

### v1 (Edge Refinement) ✅
✅ **Phase 1**: Landmark-Based Axis Estimation  
✅ **Phase 2**: Sobel Edge Detection Core  
✅ **Phase 3**: Sub-Pixel Refinement & Quality Scoring  
✅ **Phase 4**: Method Comparison & CLI Integration  
✅ **Phase 5**: Debug Visualization (12-image pipeline)  
🔄 **Phase 6**: Validation & Documentation (In Progress)

All core features are complete and functional!

## 🔧 Command-Line Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--input` | ✅ | - | Path to input image (JPG/PNG) |
| `--output` | ✅ | - | Path for output JSON file |
| `--debug` | ❌ | None | Path for debug visualization image |
| `--save-intermediate` | ❌ | False | Save intermediate processing artifacts |
| `--finger-index` | ❌ | auto | Finger to measure: auto, index, middle, ring, pinky |
| `--confidence-threshold` | ❌ | 0.7 | Minimum confidence threshold (0.0-1.0) |
| **v1 Options** | | | |
| `--edge-method` | ❌ | auto | Edge detection method: auto, contour, sobel, compare |
| `--sobel-threshold` | ❌ | 15.0 | Minimum gradient magnitude for Sobel edge detection |
| `--sobel-kernel-size` | ❌ | 3 | Sobel kernel size: 3, 5, or 7 |
| `--no-subpixel` | ❌ | False | Disable sub-pixel edge refinement |

## 📦 Dependencies

- **opencv-python** (≥4.8.0): Image processing and computer vision
- **numpy** (≥1.24.0): Numerical operations and array handling
- **mediapipe** (≥0.10.0): Hand landmark detection (pretrained model)
- **scipy** (≥1.11.0): Geometric calculations and PCA

## 📝 Important Notes

### What This Measures

The system measures the **external horizontal width** (outer diameter) of the finger at the ring-wearing zone. This is:
- ✅ The width of soft tissue + bone at the ring-wearing position
- ❌ NOT the inner diameter of a ring
- ℹ️ A geometric proxy used for downstream ring size mapping

### Current Limitations

- **Single Image**: Requires one well-composed photo (no video or multi-image support)
- **Flat Surface**: Finger and card must be on the same plane
- **Ring Size Mapping**: Does not convert to US/EU ring sizes (v1.0 scope)
- **Pretrained Models**: Uses MediaPipe's pretrained hand detector (no custom training)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MediaPipe team for the excellent hand landmark detection model
- OpenCV community for robust computer vision tools
- ISO/IEC 7810 standard for credit card dimensions reference

## 📧 Contact

For questions, issues, or feedback, please open an issue on GitHub.

---

**Made with ❤️ for accurate ring sizing**
