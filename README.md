## DX-APP (DX Application)             

**DX-APP** is DEEPX User's Application Templates based on DEEPX devices.    

This is an application examples that gives you a quick experience of NPU Accelerator performance.     
You can refer to **DX-APP** and modify it a little or implement application depending on the purpose of your use.       
This can reduce stress, such as setting the environment and implementing the code.    
Application performance may also depending on the specifications of the host because it includes pre/post processing and graphics processing operations.           
    ``` Note. Required libraries : OpenCV and dxrt. ```         

For detailed guides on running demo applications and templates, refer to the following documents:

- Overview: [01_DXNN_Application_Overview.md](./docs/source/docs/01_DXNN_Application_Overview.md)
- Installation and Build: [02_DX-APP_Installation_and_Build.md](./docs/source/docs/02_DX-APP_Installation_and_Build.md)
- Demo Guide: [03_Demo_Guide.md](./docs/source/docs/03_Demo_Guide.md)
- Template Guide: [04_Template_Guide.md](./docs/source/docs/04_Template_Guide.md)
- Classification Template: [05_Classification_Template.md](./docs/source/docs/05_Classification_Template.md)
- Object Detection Template: [06_Object_Detection_Template.md](./docs/source/docs/06_Object_Detection_Template.md)
- Python Examples: [07_Python_Examples.md](./docs/source/docs/07_Python_Examples.md)
- Appendix ChangeLog: [Appendix_Change_Log.md](./docs/source/docs/Appendix_Change_Log.md)

## Quick Start     
### prerequisites    
Install the followings first.            
- Set up build Environment ([Link](https://github.com/DEEPX-AI/dx_rt/blob/main/docs/Installation.md))     
- Install dxrt library and driver ([Link](https://github.com/DEEPX-AI/dx_rt/blob/main/docs/Getting-Started.md))     

### Installation    
- **DX_RT Drivers**        
  After installing the dxrt driver, the PCIe driver and the runtime driver. This can be verified with the lsmod command.        
  ```shell
  lsmod | grep dx
  # dxrt_driver       36864  2
  # dx_dma           139264  5 dxrt_driver
  ```
- **DX_RT library**                   
  After building, the runtime library and headers will be installed in /usr/local/lib and /usr/local/include                
  You can also modify your runtime directory in [cmake/toolchain.xx.cmake](cmake/toolchain.x86_64.cmake)     
  ```Makefile
  ...
  set(DXRT_INSTALLED_DIR /usr/local)
  ...
  ```          
- **Clone DX_APP**         
  clone this project, and just build 
  ```shell
  git clone git@gh.deepx.ai:DEEPX-AI/dx_app.git      
  ```                                 
- **Install Dependencies**                
  Install Dependency package tools             
  ```shell
  ./install.sh --dep
  ```
- **Install OpenCV (version 4.5.5 is recommended)**         
  **OpenCV** version **4.2.0** or higher is required for this demo to work properly. Please ensure your environment.
  Install and build OpenCV        
  ```shell
  ./install.sh --opencv  
  ```            
  To specify the compilation environment as aarch64 or riscv64, use the **--arch** option.     
  ```shell
  ./install.sh --arch aarch64 --opencv
  # or
  # ./install.sh --arch riscv64 --opencv
  ```           
  If you want to use the Host PC's CPU and GPU acceleration, you need to build and install OpenCV manually.    
  This process is described at [install.sh](install.sh#L134). 
  Modify the necessary flag option such as `TBB`, `IPP`, `CUDA`, etc., to **ON**, then build and install OpenCV.
### Build DX-APP    
- **Build Application**          
  ```shell
  ./build.sh 
  ```
  If you want a clean build, use the **--clean** option.          
  ```shell
  ./build.sh --clean
  ```
  To specify the compilation environment as aarch64 or riscv64, use the **--arch** option.     
  ```shell
  ./build.sh --arch aarch64
  # or
  # ./build.sh --arch riscv64
  ```              
                      
### Run Examples          
- To run the application, please refer to the script.       
  **ImageNet Classification**         
  ```shell 
  $ ./scripts/run_classifier.sh
  ```                 
  **Yolov5-s-512 ObjectDetection**         
  ```shell 
  $ ./scripts/run_detector.sh
  ```                  
- If error while loading shared library (libdxrt.so), Try to update Library Cache.      
  ```shell
  # copy your library to /usr/local/lib
  # update the system's library cache.

  sudo cp your_library.so /usr/local/lib
  sudo ldconfig 
  ```   

### ⚠️ Additional Tip
You can download some available model files and video files. Simply run the following command:

```shell
./setup.sh
```

### ⚠️ Convert Markdown Files in docs/ to PDF Using MkDocs
To convert the files under the docs/ directory into PDF using MkDocs, first build the site with:  

```shell
mkdocs build --clean
```
You can also preview the documentation as HTML in your browser using a local development server:

```shell
mkdocs serve
```
This will start a local server (usually at http://127.0.0.1:8000) where you can browse the site in real time.

Note: MkDocs alone does not support PDF export directly. 
To generate PDF files, consider using plugins such as mkdocs-pdf-export-plugin 
or convert the built HTML pages in the site/ directory to PDF using tools like wkhtmltopdf or weasyprint.

