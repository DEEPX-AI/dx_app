## DX-APP (DXNN V2)             

**DX-APP** is DEEPX User's Application Templates based on DEEPX devices.    

This is an application examples that gives you a quick experience of NPU Accelerator performance.     
You can refer to **DX-APP** and modify it a little or implement application depending on the purpose of your use.       
This can reduce stress, such as setting the environment and implementing the code.    
Application performance may also depending on the specifications of the host because it includes pre/post processing and graphics processing operations.           
    ``` Note. Required libraries : OpenCV and dxrt. ```         
## Quick Start     
### prerequisites    
Install the followings first.            
- Set up build Environment ([Link](https://github.com/DEEPX-AI/dx_rt/blob/main/dx_rt/docs/Installation.md))     
- Install dxrt library and driver ([Link](https://github.com/DEEPX-AI/dx_rt/blob/main/dx_rt/docs/Getting-Started.md))     
- model compiled for DEEPX's NPU chips ([Link](https://deepx.ai/model-zoo/))   
### Installation    
- **Dxnnv2 Drivers**        
  After installing the dxnnv2 driver, the PCIe driver and the runtime driver. This can be verified with the lsmod command.        
  ```shell
  lsmod | grep dx
  # dxrt_driver       36864  2
  # dx_dma           139264  5 dxrt_driver
  ```
- **Dxnnv2 library**                   
  After building, the runtime library and headers will be installed in /usr/local/lib and /usr/local/include                
  You can also modify your runtime directory in [cmake/toolchain.xx.cmake](/dx_app/cmake/toolchain.x86_64.cmake)     
  ```Makefile
  ...
  set(DXRT_DIR /usr/local)
  ...
  ```          
- **Clone dx_app v2**         
  clone this project, and just build 
  ```shell
  git clone git@github.com:KOMOSYS/dx_app.git      
  ```                                 
- **Install Dependencies**                
  Install Dependency package tools             
  ```shell
  ./install.sh --dep
  ```
- **Install OpenCV (version 4.5.5 is recommended)**         
  Install and build OpenCV        
  ```shell
  ./install.sh --opencv  
  ```            
  To specify the compilation environment as arm64 or riscv64, use the **--arch** option.     
  ```shell
  ./install.sh --arch arm64 --opencv
  # or
  # ./install.sh --arch riscv64 --opencv
  ```           
- **(Optional) Install onnxruntime (version 1.16.3 is recommended)**         
  Install onnxruntime library       
  ```shell
  ./install.sh --onnxruntime  
  ```        
  To specify the compilation environment as arm64, use the **--arch** option.     
  ```shell
  ./install.sh --arch arm64 --onnxruntime
  ```           
  You can also refer to [onnxruntime Build Guide](https://onnxruntime.ai/docs/build/inferencing.html).           
### Build DX-APP    
- **Build Application**          
  ```shell
  ./build.sh 
  ```
  If you want a clean build, use the **--clean** option.          
  ```shell
  ./build.sh --clean
  ```
  To specify the compilation environment as arm64 or riscv64, use the **--arch** option.     
  ```shell
  ./build.sh --arch arm64
  # or
  # ./build.sh --arch riscv64
  ```              
                      
### Run Examples          
- To run the application, please refer to the script.       
  **ImageNet Classification**         
  ```shell 
  sudo ./scripts/run_classifier.sh
  ```                 
  **Yolov5-s-512 ObjectDetection**         
  ```shell 
  sudo ./scripts/run_detector.sh
  ```                  
  [Here](/dx_app/demos/README.md) For details to run demo applications and [Here](/dx_app/templates/README.md) to run templates.         

