# 05 - Landmark-Based Axis Estimation

Module: `src/geometry.py`

## Purpose
Estimate finger axis for zone localization and cross-section measurement.

## Current behavior
- Preferred method: landmark-based axis from 4 finger landmarks (MCP/PIP/DIP/TIP).
- Multiple calculation strategies are supported internally (default favors robust fit).
- Landmark quality is validated before use.
- Automatic fallback to PCA-based axis when landmark quality is insufficient.

## Ring zone localization
- Supports landmark-aware localization mode in v1 flow.
- Legacy percentage-based zone localization remains available for compatibility.

## Failure points
- `axis_estimation_failed`
- `zone_localization_failed`

## Notes
This axis output is consumed by both contour and Sobel measurement branches.
