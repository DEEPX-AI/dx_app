#!/usr/bin/env python3
"""
Setup script for dx_postprocess pybind11 module

This uses pyproject.toml for configuration.
setup.py is kept for build configuration only.
"""

import os
import sys
from pathlib import Path

# pybind11은 pyproject.toml의 build-system.requires에서 자동 설치됨
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from setuptools import setup
except ImportError:
    print("ERROR: pybind11 not found.")
    print("It should be auto-installed via pyproject.toml")
    print("Try: pip install --upgrade pip setuptools")
    sys.exit(1)

# Source files (relative paths only)
sources = [
    "yolo_post_processing.cpp",
    "yolo_post_processing_pybinding.cpp"
]

# Check if source files exist
for src in sources:
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source file not found: {src}")

# 빌드 타입 감지
build_type = os.environ.get('CMAKE_BUILD_TYPE', 'Release').lower()
debug_mode = os.environ.get('DEBUG', '0') == '1'

print(f"\n=== dx_postprocess Build Configuration ===")
print(f"  CMAKE_BUILD_TYPE: {os.environ.get('CMAKE_BUILD_TYPE', 'Not set')}")
print(f"  DEBUG: {os.environ.get('DEBUG', 'Not set')}")
print(f"  Detected build_type: {build_type}")
print(f"  Detected debug_mode: {debug_mode}")

# 컴파일러 플래그 설정
extra_compile_args = []
extra_link_args = []

if build_type == 'debug' or debug_mode:
    extra_compile_args.extend(['-g', '-O0', '-DDEBUG'])
    extra_link_args.extend(['-g'])
    print(f"  Build mode: DEBUG")
else:
    extra_compile_args.extend(['-O3', '-DNDEBUG'])
    print(f"  Build mode: RELEASE")

print(f"  Compile flags: {extra_compile_args}")
if extra_link_args:
    print(f"  Link flags: {extra_link_args}")
print("=" * 45 + "\n")

# Define the extension
ext_modules = [
    Pybind11Extension(
        "dx_postprocess",
        sources,
        include_dirs=["."],
        cxx_std=14,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

class CustomBuildExt(build_ext):
    """Custom build extension with verbose output"""
    
    def build_extensions(self):
        print("Building dx_postprocess module...")
        print(f"  Python: {sys.executable}")
        print(f"  Platform: {sys.platform}")
        
        try:
            import pybind11
            print(f"  pybind11: {pybind11.__version__}")
        except ImportError as e:
            print(f"  Warning: Cannot verify pybind11 version: {e}")
        
        super().build_extensions()
        print("Build completed successfully!\n")

# setup() with minimal config - most comes from pyproject.toml
setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
)