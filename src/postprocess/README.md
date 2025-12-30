# Postprocess Libraries Build System

This directory contains an independent build system for the DEEPX postprocess libraries. It can be used to build all postprocess libraries without building the entire dx_app project.

## Overview

The postprocess libraries include:
- **scrfd_postprocess**: Face detection postprocessing
- **yolov5_postprocess**: YOLOv5 object detection postprocessing  
- **yolov5face_postprocess**: YOLOv5 face detection postprocessing
- **yolov5pose_postprocess**: YOLOv5 pose estimation postprocessing
- **yolov7_postprocess**: YOLOv7 object detection postprocessing
- **yolov8_postprocess**: YOLOv8 object detection postprocessing
- **yolov9_postprocess**: YOLOv9 object detection postprocessing
- **yolox_postprocess**: YOLOX object detection postprocessing

## Prerequisites

Before building, ensure you have:

1. **CMake** (version 3.16 or higher)
2. **C++ compiler** with C++17 support (GCC 7+, Clang 6+)
3. **dxrt library** installed and accessible
4. **pkg-config** (recommended)
5. **Build system**: Make or Ninja

### Installing dxrt

The dxrt library is required for building. Make sure it's installed and can be found by:
- pkg-config (`pkg-config --exists dxrt`)
- Or available in the system library path

## Quick Start

### Using Makefile (Recommended)

```bash
# Show available commands
make help

# Build all libraries in Release mode
make build

# Build in Debug mode
make debug

# Clean and rebuild
make clean

# Show current build status
make status
```

### Using Build Script Directly

```bash
# Basic build (Release mode)
./build_postprocess.sh

# Build in Debug mode
./build_postprocess.sh --type Debug

# Clean build
./build_postprocess.sh --clean

# Verbose output
./build_postprocess.sh --verbose

# Set custom build directory
./build_postprocess.sh --build-dir my_build

# Set installation prefix
./build_postprocess.sh --prefix /usr/local
```

### Using CMake Directly

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
make -j$(nproc)
```

## Build Script Options

The `build_postprocess.sh` script supports the following options:

| Option | Description | Default |
|--------|-------------|---------|
| `-t, --type TYPE` | Build type (Debug\|Release\|RelWithDebInfo\|MinSizeRel) | Release |
| `-d, --build-dir DIR` | Build directory | ./build |
| `-c, --clean` | Clean build directory before building | false |
| `-p, --prefix DIR` | Installation prefix directory | (none) |
| `-v, --verbose` | Enable verbose output | false |
| `-h, --help` | Show help message | - |

## Output Structure

After a successful build, you'll find:

```
build/
├── lib/                           # Shared libraries (.so files)
│   ├── libscrfd_postprocess.so
│   ├── libscrfd_postprocess.so.1
│   ├── libyolov5_postprocess.so
│   ├── libyolov5_postprocess.so.1
│   ├── libyolov5face_postprocess.so
│   ├── libyolov5face_postprocess.so.1
│   ├── libyolov5pose_postprocess.so
│   ├── libyolov5pose_postprocess.so.1
│   ├── libyolov7_postprocess.so
│   ├── libyolov7_postprocess.so.1
│   ├── libyolov8_postprocess.so
│   ├── libyolov8_postprocess.so.1
│   ├── libyolov9_postprocess.so
│   ├── libyolov9_postprocess.so.1
│   ├── libyolox_postprocess.so
│   └── libyolox_postprocess.so.1
└── include/                       # Header files
    ├── common_util.hpp
    ├── common_util_inline.hpp
    ├── scrfd_postprocess.h
    ├── yolov5_postprocess.h
    ├── yolov5face_postprocess.h
    ├── yolov5pose_postprocess.h
    ├── yolov7_postprocess.h
    ├── yolov8_postprocess.h
    ├── yolov9_postprocess.h
    └── yolox_postprocess.h
```

## Usage Examples

### Basic Build
```bash
# Simple release build
make build
```

### Development Build
```bash
# Debug build with verbose output
./build_postprocess.sh --type Debug --verbose
```

### Production Build and Install
```bash
# Build and install to system
./build_postprocess.sh --type Release --prefix /usr/local
```

### Multiple Build Types
```bash
# Build all types for testing
make build-all-types
```

## Linking Against Libraries

To use these libraries in your project:

### CMake
```cmake
find_library(YOLOV5_POSTPROCESS_LIB yolov5_postprocess PATHS /path/to/build/lib)
target_link_libraries(your_target ${YOLOV5_POSTPROCESS_LIB})
```

### pkg-config (if installed)
```bash
pkg-config --libs --cflags yolov5_postprocess
```

### Direct linking
```bash
g++ -I/path/to/build/include -L/path/to/build/lib -lyolov5_postprocess your_code.cpp
```

## Troubleshooting

### Common Issues

1. **dxrt library not found**
   ```
   Solution: Install dxrt or set LD_LIBRARY_PATH to include dxrt location
   ```

2. **CMake version too old**
   ```
   Solution: Update CMake to version 3.16 or higher
   ```

3. **Compiler errors**
   ```
   Solution: Ensure C++17 compatible compiler (GCC 7+, Clang 6+)
   ```

4. **Missing dependencies**
   ```
   Solution: Install pkg-config and ensure all system dependencies are available
   ```

### Debug Build Issues

If you encounter issues:

1. Try a clean build:
   ```bash
   make clean
   ```

2. Use verbose output:
   ```bash
   ./build_postprocess.sh --verbose
   ```

3. Check dependencies:
   ```bash
   ldd build/lib/libyolov5_postprocess.so
   ```

## Integration with Main Project

This build system is designed to work independently, but the generated libraries are compatible with the main dx_app project. You can:

1. Build postprocess libraries separately for faster development
2. Use the generated libraries in the main project
3. Install libraries system-wide for use in other projects

## License

See the main project LICENSE file for license information.