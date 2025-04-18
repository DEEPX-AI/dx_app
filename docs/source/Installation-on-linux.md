## Prerequisites

refer to the followings first. 

- Set up the build Environment ([Link](https://github.com/DEEPX-AI/dx_rt/blob/main/docs/Installation.md))     
- Install the dxrt library and driver ([Link](https://github.com/DEEPX-AI/dx_rt/blob/main/docs/Getting-Started.md))     
- Obtain a model compiled for DEEPX's NPU chips ([Link](https://deepx.ai/model-zoo/))   

## DX Runtime Setup

**DX_RT Drivers**

After installing the dxrt driver, both the PCIe driver and the runtime driver should be available.
You can verify the installation using the `lsmod` command:

```shell
lsmod | grep dx
# dxrt_driver            53248  2
# dx_dma                475136  7 dxrt_driver
```

**DX_RT Runtime library**

After building, the runtime library and headers will be installed in /usr/local/lib and /usr/local/include.
You can also modify your runtime directory in *cmake/toolchain.x86_64.cmake*.

```cmake
...
set(DXRT_INSTALLED_DIR /usr/local)
...
```

## DX Application Setup 

**Installation and Build On Linux**

(on Linux) The dx_app requires the following components to be installed with the specified versions or later :

- dx_rt_npu_linux_driver (v1.3.3 or later)
- dx_rt (v2.7.0 or later)
- dx_fw (v1.6.3 or later)

All necessary modules can be found in DEEPX-AI.

**Installation Options**

Run the following command to check available installation options: 

```shell
./install.sh --help
```

Available options: 
```shell
  --help show this help
  --arch target CPU architecture : [ x86_64, aarch64, riscv64 ]
  --dep install dependencies : cmake, gcc, ninja, etc..
  --opencv (optional) install opencv pkg
  --opencv-source-build (optional) install opencv pkg by source build
  --all install dependencies & opencv pkg
```

If you want to enable CPU/GPU acceleration, OpenCV must be built and installed manually. 
Modify the necessary flag options (e.g., TBB, IPP, CUDA) to ON, then build and install OpenCV. 

If OpenCV is already set up, manually configure OpenCV_DIR in the cmake/toolchain.xxx.cmake file:

```cmake
  set(CMAKE_SYSTEM_NAME Linux)
  set(CMAKE_SYSTEM_PROCESSOR x86_64)
  set(DXRT_INSTALLED_DIR /usr/local)
  set(OpenCV_DIR /your/opencv/installation/dir)
  set(onnxruntime_LIB_DIRS /usr/local/lib)
```

**Building and Running dx_app**

To build dx_app

```shell
./build.sh ## Use --clean for a clean build
```

To download required models and videos, run : 

```shell
./setup.sh
```
This will install models and videos under the `assets/` folder.
the available models include Classification, Object Detection, and Segmentation models.

To test dx_app, run :

```shell
./scripts/run_detector.sh
```

**Handling shared Library Error**

If you encounter an error while loading the shared library (`libdxrt.so`), try updating the library cache:

```shell
# Copy your library to /usr/local/lib
sudo cp your_library.so /usr/local/lib

# Update the system's library cache
sudo ldconfig
```

For more details on running demo applications, refer to the dx_app/demos applications.

For information on running templates, check the dx_app/templates applications.

