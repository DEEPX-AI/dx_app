#!/usr/bin/env python3
"""
Setup script for dx_postprocess pybind11 module
"""

import os
import sys
import subprocess
from pathlib import Path

# Check if pybind11 exists, if not download it
pybind11_dir = Path(__file__).parent.parent.parent / "extern" / "pybind11"

if not pybind11_dir.exists():
    print(f"Downloading pybind11 to {pybind11_dir}...")
    pybind11_dir.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        "git", "clone", "--branch", "v2.12.0", "--depth", "1",
        "https://github.com/pybind/pybind11.git", str(pybind11_dir)
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Failed to download pybind11: {result.stderr}")
        sys.exit(1)
    print("pybind11 downloaded successfully!")

# Add pybind11 to Python path
sys.path.insert(0, str(pybind11_dir))

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from setuptools import setup
except ImportError as e:
    print(f"Failed to import pybind11: {e}")
    print(f"pybind11 directory: {pybind11_dir}")
    print(f"pybind11 exists: {pybind11_dir.exists()}")
    if pybind11_dir.exists():
        print(f"pybind11 contents: {list(pybind11_dir.iterdir())}")
    sys.exit(1)

# Source files (relative paths only)
sources = [
    "yolo_post_processing.cpp",
    "yolo_post_processing_pybinding.cpp"
]

# Check if source files exist
for src in sources:
    if not os.path.exists(src):
        print(f"Error: Source file {src} not found!")
        sys.exit(1)

# Define the extension
ext_modules = [
    Pybind11Extension(
        "dx_postprocess",
        sources,
        include_dirs=["."],  # Add current directory to include path
        cxx_std=14,
    ),
]

class CustomBuildExt(build_ext):
    """Custom build extension"""
    
    def build_extensions(self):
        print("Building dx_postprocess module...")
        print(f"Python: {sys.executable}")
        print(f"Platform: {sys.platform}")
        print(f"pybind11 path: {pybind11_dir}")
        
        try:
            import pybind11
            print(f"pybind11: {pybind11.__version__}")
        except ImportError as e:
            print(f"Warning: {e}")
        
        super().build_extensions()

setup(
    name="dx_postprocess",
    version="0.1.0",
    author="DEEPX",
    author_email="yjsong@deepx.ai",
    description="YOLO post-processing module for DEEPX applications",
    long_description="Python binding for YOLO post-processing functions used in DEEPX applications",
    long_description_content_type="text/plain",
    python_requires=">=3.7",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
) 