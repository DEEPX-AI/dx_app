## DX-APP (DXNN V2)             
**DX-APP V2** is DEEPX User's Application Templates based on DEEPX devices.    
    ``` Note. This project cannot be used without OpenCV ```         

## Quick Start     
### prerequisites    
- dxnnv2 library and driver     
  ``` Note. Please contact DEEPX for runtime library and driver module ```         
- model compiled for DEEPX's NPU chips      
### Installation    
- **Dxnnv2 Drivers**        
  After installing the dxnnv2 driver, the PCIe driver and the runtime driver. This can be verified with the lsmod command.        
  ```shell
  cd dx_rt/driver
  ./install_m1.sh
  lsmod | grep dx
  # dxrt_driver       36864  2
  # dx_dma           139264  5 dxrt_driver
  ```
- **Dxnnv2 library**           
  Install to your path using the --install option         
  After building, the runtime library and headers will be installed in /usr/local/lib and /usr/local/include
  ```shell
  cd dx_rt
  ./build.sh --install /usr/local 
  ```            
  To specify the compilation environment as arm64 or riscv64, use the **--arch** option.
  ```shell
  ./build.sh --install /usr/local --arch arm64
  # or 
  # ./build.sh --install /usr/local --arch riscv64 
  ```
  If you want a clean build, use the **--clean** option.          
  ```shell
  ./build.sh --clean --install /usr/local
  ```
- **Clone dx_app v2**         
  clone this project, and just build 
  ```shell
  git clone git@github.com:KOMOSYS/dx_app.git      
  ```                                 
- **Install Dependencies**                
  Install Dependencies              
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

             
          
         
