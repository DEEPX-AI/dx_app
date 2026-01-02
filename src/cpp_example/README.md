# C++ Examples Build System

This directory contains an independent build system for the DEEPX C++ examples. It can build all examples independently, with optional integration to the postprocess libraries.

## Overview

The C++ examples are organized into categories:

### Input Source Processing Examples
- **image_test**: Basic image processing with DXNN models
- **image_multi_model_test**: Multi-model image processing
- **image_libjpeg_test**: JPEG image processing with libjpeg
- **camera_test**: Camera input processing
- **camera_v4l2_test**: V4L2 camera interface example

### Classification Examples
- **efficientNet_example**: EfficientNet classification (sync)
- **efficientNet_async_example**: EfficientNet classification (async)

### Multi-Channel Examples
- **multi_channel_yolov5s_example**: Multi-channel YOLOv5s processing

### Object Detection Examples (requires postprocess libraries)
- **scrfd_example_sync/async**: SCRFD face detection
- **yolov5_example_sync/async**: YOLOv5 object detection
- **yolov5face_example_sync/async**: YOLOv5 face detection
- **yolov5pose_example_sync/async**: YOLOv5 pose estimation
- **yolov7_example_sync/async**: YOLOv7 object detection
- **yolov8_example_sync/async**: YOLOv8 object detection
- **yolov9_example_sync/async**: YOLOv9 object detection
- **yolox_example_sync/async**: YOLOX object detection

## Prerequisites

### Required Dependencies
1. **CMake** (version 3.16 or higher)
2. **C++ compiler** with C++14 support (GCC 4.9+, Clang 3.4+)
3. **OpenCV** (for image processing)
4. **dxrt library** (DEEPX Runtime)
5. **pkg-config** (recommended)
6. **Build system**: Make or Ninja

### Optional Dependencies
- **Postprocess libraries**: Required only for object detection examples
  - Build with: `cd ../postprocess && ./build_postprocess.sh`

## Quick Start

### Using Makefile (Recommended)

```bash
# Show available commands
make help

# Build basic examples (no postprocess libs needed)
make build

# Build postprocess libraries first, then all examples
make postprocess-libs

# Build everything
make all

# Show current build status
make status
```

### Using Build Script Directly

```bash
# Basic build (only examples that don't need postprocess libs)
./build_examples.sh

# Build postprocess libraries first, then all examples
./build_examples.sh --build-postprocess

# Build in Debug mode with verbose output
./build_examples.sh --type Debug --verbose

# Clean build
./build_examples.sh --clean
```

### Using CMake Directly

```bash
# Create build directory
mkdir build && cd build

# Configure (basic examples only)
cmake ..

# Configure with postprocess libraries
cmake -DPOSTPROCESS_LIB_DIR=../postprocess/build/lib \
      -DPOSTPROCESS_INCLUDE_DIR=../postprocess/build/include ..

# Build
make -j$(nproc)
```

## Build Script Options

The `build_examples.sh` script supports:

| Option | Description | Default |
|--------|-------------|---------|
| `-t, --type TYPE` | Build type (Debug\|Release\|RelWithDebInfo\|MinSizeRel) | Release |
| `-d, --build-dir DIR` | Build directory | ./build |
| `-c, --clean` | Clean build directory before building | false |
| `--build-postprocess` | Build postprocess libraries first | false |
| `--postprocess-lib DIR` | Path to postprocess libraries | ../postprocess/build/lib |
| `--postprocess-inc DIR` | Path to postprocess headers | ../postprocess/build/include |
| `-v, --verbose` | Enable verbose output | false |
| `-h, --help` | Show help message | - |

## Build Modes

### Basic Mode (No Postprocess Libraries)
If postprocess libraries are not available, the build system will create:
- Input processing examples
- Classification examples
- Multi-channel examples (without postprocess integration)

### Full Mode (With Postprocess Libraries)
When postprocess libraries are available, additionally builds:
- All object detection examples (sync and async versions)
- Full multi-channel integration

## Output Structure

After a successful build:

```
build/
└── bin/                           # All example executables
    ├── image_test                 # Basic image processing
    ├── image_multi_model_test     # Multi-model processing
    ├── camera_test               # Camera input
    ├── efficientNet_example      # Classification
    ├── multi_channel_yolov5s_example  # Multi-channel
    ├── scrfd_example_sync        # Object detection (if postprocess available)
    ├── scrfd_example_async
    ├── yolov5_example_sync
    ├── yolov5_example_async
    └── ...                       # Other detection examples
```

## Usage Examples

### Input Processing
```bash
# Basic image test
./build/bin/image_test -m model.dxnn -i image.jpg

# Camera test
./build/bin/camera_test -m model.dxnn -d 0

# Multi-model test
./build/bin/image_multi_model_test -m1 model1.dxnn -m2 model2.dxnn -i image.jpg
```

### Classification
```bash
# EfficientNet classification
./build/bin/efficientNet_example -m efficientnet.dxnn -i image.jpg

# Async classification
./build/bin/efficientNet_async_example -m efficientnet.dxnn -i image.jpg -l 1000
```

### Object Detection (requires postprocess libraries)
```bash
# YOLOv5 synchronous detection
./build/bin/yolov5_example_sync -m yolov5.dxnn -i image.jpg

# YOLOv5 asynchronous detection
./build/bin/yolov5_example_async -m yolov5.dxnn -i image.jpg -l 1000 --no-display

# SCRFD face detection
./build/bin/scrfd_example_sync -m scrfd.dxnn -i image.jpg

# YOLOX detection
./build/bin/yolox_example_async -m yolox.dxnn -i image.jpg -l 5000 --no-display
```

### Multi-Channel Processing
```bash
# Multi-channel YOLOv5s
./build/bin/multi_channel_yolov5s_example -m yolov5s.dxnn -i "image1.jpg,image2.jpg"
```

## Common Command Options

Most examples support these options:

| Option | Description | Example |
|--------|-------------|---------|
| `-m, --model` | Path to DXNN model file | `-m model.dxnn` |
| `-i, --input` | Input image/video path | `-i image.jpg` |
| `-d, --device` | Camera device number | `-d 0` |
| `-l, --loop` | Number of inference loops | `-l 1000` |
| `--no-display` | Disable result display | `--no-display` |
| `-o, --output` | Output path for results | `-o result.jpg` |

## Workflow Examples

### Development Workflow
```bash
# 1. Build basic examples for testing
make build

# 2. Test basic functionality
./build/bin/image_test -m your_model.dxnn -i test_image.jpg

# 3. Build postprocess libraries and full examples
make postprocess-libs

# 4. Test object detection
./build/bin/yolov5_example_sync -m yolov5_model.dxnn -i test_image.jpg
```

### Production Build
```bash
# Clean build with all features
./build_examples.sh --build-postprocess --clean --type Release

# Install to system
./build_examples.sh --build-postprocess --prefix /usr/local
```

## Integration with Postprocess Libraries

The build system automatically detects and uses postprocess libraries:

1. **Auto-detection**: Looks for libraries in `../postprocess/build/lib`
2. **Custom paths**: Use `--postprocess-lib` and `--postprocess-inc` options
3. **Auto-build**: Use `--build-postprocess` to build libraries automatically

## Troubleshooting

### Common Issues

1. **OpenCV not found**
   ```bash
   # Install OpenCV development packages
   sudo apt-get install libopencv-dev
   ```

2. **dxrt library not found**
   ```bash
   # Ensure dxrt is installed and in library path
   export LD_LIBRARY_PATH=/path/to/dxrt/lib:$LD_LIBRARY_PATH
   ```

3. **Postprocess libraries not found**
   ```bash
   # Build postprocess libraries first
   cd ../postprocess && ./build_postprocess.sh
   ```

4. **CMake version too old**
   ```bash
   # Update CMake to 3.16+
   ```

### Debug Build Issues

```bash
# Clean build with debug info
make clean-debug

# Verbose build
./build_examples.sh --verbose --type Debug

# Check dependencies
ldd build/bin/yolov5_example_sync
```

## Integration with Main Project

This build system works independently but is compatible with the main dx_app project:

1. **Independent development**: Build and test examples separately
2. **Shared dependencies**: Uses same dxrt and OpenCV dependencies
3. **Library integration**: Seamlessly uses postprocess libraries when available

## Performance Notes

- **Release builds**: Use `-O3` optimization for best performance
- **Debug builds**: Include debug symbols for development
- **Async examples**: Use multiple threads for better throughput
- **Memory usage**: Monitor with tools like `valgrind` for large datasets

## License

See the main project LICENSE file for license information.