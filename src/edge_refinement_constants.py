"""
Constants for Sobel edge refinement algorithm.

This module contains all configurable parameters and thresholds used
in the edge refinement pipeline to make them easy to tune and maintain.
"""

# =============================================================================
# ROI Extraction Constants
# =============================================================================

# ROI padding around zone for gradient context
ROI_PADDING_PX = 50

# Finger width estimation factor (conservative to ensure full capture)
# Typical finger aspect ratio is 3:1 to 5:1 (length:width)
FINGER_WIDTH_RATIO = 3.0  # length / width


# =============================================================================
# Sobel Filter Constants
# =============================================================================

# Default Sobel kernel size
DEFAULT_KERNEL_SIZE = 3

# Valid kernel sizes
VALID_KERNEL_SIZES = [3, 5, 7]


# =============================================================================
# Edge Detection Constants
# =============================================================================

# Default gradient threshold for valid edge
DEFAULT_GRADIENT_THRESHOLD = 15.0

# Realistic finger width range for validation
# Based on typical adult finger widths across ring sizes
MIN_FINGER_WIDTH_CM = 1.6  # Size 6 (16mm)
MAX_FINGER_WIDTH_CM = 2.5  # Size 13 (23mm)

# Tolerance for expected width comparison (when contour available)
WIDTH_TOLERANCE_FACTOR = 0.25  # ±25%


# =============================================================================
# Sub-Pixel Refinement Constants
# =============================================================================

# Maximum sub-pixel refinement offset from integer position
MAX_SUBPIXEL_OFFSET = 0.5  # ±0.5 pixels

# Minimum denominator value to avoid division by zero in parabola fitting
MIN_PARABOLA_DENOMINATOR = 1e-6


# =============================================================================
# Outlier Filtering Constants
# =============================================================================

# MAD (Median Absolute Deviation) threshold multiplier
MAD_OUTLIER_THRESHOLD = 3.0  # Outliers are >3 MAD from median


# =============================================================================
# Edge Quality Scoring Constants
# =============================================================================

# Gradient strength normalization (typical strong edge magnitude)
GRADIENT_STRENGTH_NORMALIZER = 30.0

# Smoothness scoring (variance to exponential mapping)
SMOOTHNESS_VARIANCE_NORMALIZER = 200.0

# Quality score component weights
QUALITY_WEIGHT_GRADIENT = 0.4  # Gradient strength: 40%
QUALITY_WEIGHT_CONSISTENCY = 0.3  # Edge consistency: 30%
QUALITY_WEIGHT_SMOOTHNESS = 0.2  # Edge smoothness: 20%
QUALITY_WEIGHT_SYMMETRY = 0.1  # Bilateral symmetry: 10%


# =============================================================================
# Auto Fallback Decision Constants
# =============================================================================

# Minimum quality score to use Sobel (otherwise fall back to contour)
MIN_QUALITY_SCORE_THRESHOLD = 0.65  # Lowered from 0.7 for mask-constrained mode

# Minimum edge detection success rate
MIN_CONSISTENCY_THRESHOLD = 0.30  # 30% (lowered from 50% for mask-constrained mode)

# Realistic measurement range for validation
MIN_REALISTIC_WIDTH_CM = 0.8
MAX_REALISTIC_WIDTH_CM = 3.5

# Maximum allowed difference from contour measurement (percentage)
MAX_CONTOUR_DIFFERENCE_PCT = 50.0
