# Scripts

Utility scripts for development and testing.

## test.sh

Quick test script for running the ring-sizer measurement tool.

### Usage

```bash
# Run with auto-detected test image (with debug output)
./script/test.sh

# Run with specific image
./script/test.sh input/my_image.jpg

# Run without debug visualization (faster)
./script/test.sh --no-debug

# Show help
./script/test.sh --help
```

### Features

- Auto-detects first available test image in `input/` directory
- Automatically creates `.venv` and installs dependencies if needed
- Outputs JSON results to `output/test_result.json`
- Generates debug visualization at `output/test_debug.png`
- Pretty-prints JSON results to console
- Auto-opens debug image on macOS (if available)
- Color-coded console output for better readability

### Requirements

- Python 3.8+
- Test images in `input/` directory

## build.sh

(To be implemented)

Build and packaging scripts for deployment.
