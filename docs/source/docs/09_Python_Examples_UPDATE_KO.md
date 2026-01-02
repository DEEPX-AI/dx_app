# Python 예제 가이드

## 1. 소개

Python 예제는 DEEPX NPU를 활용하여 AI 애플리케이션을 개발하려는 사용자를 위한 시작 가이드를 제공합니다. 각 예제는 실제 모델을 기반으로 한 독립적인 End-to-End 스크립트로, 다음과 같은 목표를 가지고 설계되었습니다.

-   **모델 맞춤형의 명확한 파이프라인**: 각 예제는 특정 모델 하나만을 위한 독립적인 스크립트로, 복잡한 분기나 추상화 없이 해당 모델의 파이프라인(입력-전처리-추론-후처리)을 명확하게 보여줍니다. 이는 사용자가 최소한의 코드로 동작 원리를 파악하고, 이를 기반으로 자신의 애플리케이션을 쉽게 확장할 수 있도록 돕기 위함입니다.

-   **다양한 최적화 기법 학습**: 동일한 모델에 대해 파이프라인 동기/비동기 처리, Python/C++ 후처리 등 다양한 예제 variant를 제공합니다. 사용자는 각 예제를 비교하며 성능 최적화 기법을 자연스럽게 학습할 수 있습니다.

-   **실용적인 개발 기반 제공**: 이 예제들을 기반으로, 사용자는 자신의 애플리케이션에 맞는 적합한 구조를 설계하고 코드를 확장해 나갈 수 있는 실용적인 인사이트를 얻을 수 있습니다.

---

## 2. 사전 준비

예제를 실행하기 전에 다음 단계를 통해 필요한 환경을 설정해야 합니다.

### 2.1. Assets 다운로드

예제 실행을 위해 모델(`.dxnn`) 및 비디오 파일이 필요합니다. [`setup.sh`](../../../setup.sh) 스크립트 실행 결과, `assets/models`와 `assets/videos` 경로에 각각 샘플 모델들과 샘플 비디오들이 다운로드됩니다.   
```bash
# (dx_app/ 경로에서 실행)
./set_up.sh
```

### 2.2. `dx_engine` 파이썬 라이브러리 설치

`dx_engine`은 DEEPX NPU에서 `.dxnn` 모델 추론을 실행하기 위한 핵심 파이썬 라이브러리입니다. [DX-RT](https://github.com/DEEPX-AI/dx_rt) (DEEPX Runtime) SDK의 일부로, DX-RT C++ API의 파이썬 바인딩을 제공하여 파이썬 환경에서 NPU 가속을 사용할 수 있게 합니다. 모든 파이썬 예제는 `dx_engine`을 기반으로 동작합니다.

`dx_engine`은 아래 방법으로 설치할 수 있습니다.

```bash
# dx_rt 전체 빌드 시 함께 설치
git clone https://github.com/DEEPX-AI/dx_rt
cd dx_rt
./install.sh --all
./build.sh
```

### 2.3. `dx_postprocess` 파이썬 라이브러리 설치

[`dx_postprocess`](../../../src/bindings/python/dx_postprocess/)는 [C++ 예제](../../../src/cpp_example)에서 사용하는 [후처리 클래스](../../../src/postprocess)를 파이썬에서 사용할 수 있도록 `pybind11`로 래핑한 라이브러리입니다. Python 후처리의 성능 최적화가 필요할 때 사용되며, 파일 이름 끝에 `'_cpp_postprocess'`가 포함된 예제들이 이 라이브러리를 사용합니다.

`dx_postprocess`는 아래 방법으로 설치할 수 있습니다.

```bash
# (dx_app/ 경로에서 실행)
# 1. dx_app 전체 빌드 시 함께 설치
./build.sh

# 2. 또는 독립 설치
./src/bindings/python/dx_postprocess/install.sh
```

### 2.4. 의존성 패키지 설치
파이썬 예제 실행에 필요한 `Numpy`, `OpenCV` 등의 의존성 패키지들을 설치합니다. 아래 명령어를 실행하여 [`requirements.txt`](../../../src/python_example/requirements.txt) 파일에 명시된 패키지들을 설치하십시오.
```bash
# (dx_app/ 경로에서 실행)
pip install -r src/python_example/requirements.txt
```

---

## 3. 예제 구조

Python 예제는 사용자가 원하는 예제를 쉽게 찾고 모델 및 변형 예제 간의 차이점을 직관적으로 비교할 수 있도록 다음과 같은 구조로 설계되었습니다.

```
python_example/
├── classification/
│   └── efficientnet/
│       └── efficientnet_sync.py
├── object_detection/
│   ├── yolov5/
│   │   ├── yolov5_sync.py
│   │   ├── yolov5_sync_cpp_postprocess.py
│   │   ├── yolov5_async.py
│   │   └── yolov5_async_cpp_postprocess.py
│   └── ... (다른 모델들)
├── semantic_segmentation/
│   └── ...
└── ppu/
    └── ...
```

-   **Task 별 폴더**: `classification`, `object_detection`, `semantic_segmentation` 등 AI Task 별로 폴더가 구성되어 있습니다. `ppu` 폴더는 `.onnx` 모델 컴파일 시 PPU(Post-Processing Unit) 옵션이 적용된 `.dxnn` 모델을 활용하는 특화된 예제를 포함합니다.
-   **모델 별 폴더**: 각 Task 폴더 하위에는 `yolov5`, `efficientnet` 등 모델별로 폴더가 나뉘어 있습니다.
-   **예제 Variants**: 하나의 모델 폴더 안에는 파이프라인 처리 방식(동기/비동기)과 후처리 구현(Python/C++)에 따라 여러 변형(Variant)의 예제 스크립트가 존재합니다.

---

## 4. 예제 Variants 상세 설명

각 모델은 **파이프라인 처리 방식과 후처리 구현의 조합**에 따라 여러 Variant를 제공하며, 사용자는 스크립트 구현 방식이 어떻게 처리 성능에 영향을 미치는지 직접 비교하고 체험할 수 있습니다.


### 4.1. 파이프라인 처리 방식: Sync vs. Async

**sync 예제**와 **async 예제**는 파이프라인 구조뿐만 아니라, NPU를 사용하는 방식에서도 근본적인 차이가 있습니다. 

#### **`sync` 예제**
파일 이름에 `'_sync'`가 포함된 예제들입니다.

-   **파이프라인**: 파이프라인의 각 단계가 **하나의 스레드에서 순차적으로 처리**됩니다.
-   **NPU 처리**: Inference 단계에서 `dx_engine`의 `InferenceEngine.run()` API를 호출하며, 이 API는 **한 번에 하나의 추론 요청만** 처리합니다.
-   **장점**: 전체적인 구조가 간단하여 동작 원리를 처음 파악하기에 용이합니다.

#### **`async` 예제**
파일 이름에 `'_async'`가 포함된 예제들입니다.
-   **파이프라인**: 파이프라인의 각 단계가 **별도의 스레드에서 병렬적으로 처리**됩니다.
-   **NPU 처리**: Inference 단계에서 `dx_engine`의 `InferenceEngine.run_async()` API를 호출합니다. 이 API는 여러 추론 요청을 비동기적으로 처리하여 NPU의 유휴 시간을 최소화하고 높은 추론 처리량을 달성할 수 있습니다.
-   **장점**: 스트림(비디오, 카메라 등)과 같이 연속적인 데이터를 처리할 때 하드웨어 리소스를 효과적으로 활용하여 높은 처리량(End-to-End FPS)를 얻을 수 있습니다.

### 4.2. 후처리 구현 방식: Python vs. C++

#### **Python Post-process 예제**
파일 이름에 별도 접미사(suffix)가 없는 기본 예제들입니다.
-   **구현 방식**: 후처리 로직이 **Python 스크립트로 직접 구현**되어 있습니다.
-   **장점**: Python 코드만으로 전체 파이프라인의 동작을 쉽게 이해하고 디버깅할 수 있습니다.

#### **C++ Post-process 예제**
파일 이름에 `'_cpp_postprocess'`가 포함된 예제들입니다.
-   **구현 방식**: [`dx_postprocess`](../../../src/bindings/python/dx_postprocess/)는 [C++ 예제](../../../src/cpp_example)에서 사용하는 [모델별 후처리 클래스](../../../src/postprocess)를 `pybind11`로 래핑한 라이브러리입니다. `'_cpp_postprocess'` 예제에서는 기존의 Python 후처리 로직 대신, 이 라이브러리를 임포트하여 C++로 구현된 후처리 함수를 직접 호출합니다.
-   **장점**: 다음과 같은 상황에서 성능을 최적화하는 데 도움이 될 수 있습니다.
    1.  **후처리 연산 가속**: Python으로 구현된 후처리 연산 자체가 느려 병목이 될 때, C++ 구현으로 이를 직접 가속합니다.
    2.  **CPU 리소스 경합 완화**: 임베디드 환경과 같이 CPU 성능이 제한적인 경우, Python 후처리의 높은 CPU 사용률이 다른 파이프라인 단계를 방해할 수 있습니다. C++ 후처리는 CPU 부하를 줄여 시스템 전체의 효율을 높이고 NPU의 처리량 개선에 기여합니다.

---

## 5. 예제의 주요 기능과 실행 모드

모든 예제는 이미지 또는 스트림 입력을 받아 추론을 수행하며, 실행 완료 후 콘솔에 처리 성능 요약 정보를 출력합니다.

### 5.1. 이미지 추론 (`image_inference`)

-   **기능**: 단일 이미지에 대한 추론을 수행하고 결과를 처리합니다.
    -   기본적으로 결과 이미지를 화면에 시각화합니다. (Classification 모델은 예외)
    -   `--no-display` 옵션 사용 시, 화면에 출력하는 대신 결과 이미지를 `artifacts/python_example/<task>/` 폴더에 저장합니다.
-   **미지원 예제**: `async` 예제는 스트림 처리 성능 개선을 위한 예제이므로, `image_inference` 기능을 제공하지 않습니다.
-   **성능 리포트**: 실행 완료 후, 콘솔에 각 처리 단계의 소요 시간을 담은 **`Image Processing Summary`** 를 출력합니다.
    -   `--no-display` 옵션 사용 시, `Display` 항목은 출력에서 제외됩니다.
    <details>
    <summary> Image Processing Summary 출력 예시</summary>

    ```
    ===================================
    IMAGE PROCESSING SUMMARY      
    ===================================
    Pipeline Step       Latency
    -----------------------------------
    Read                2.04 ms
    Preprocess          1.15 ms
    Inference          39.50 ms
    Postprocess         2.63 ms
    Display             0.36 ms
    -----------------------------------
    Total Time      :   45.7 ms
    ===================================
    ```
    </details>

### 5.2. 스트림 추론 (`stream_inference`)

-   **기능**: 연속적인 프레임(비디오, RTSP, 카메라)을 입력받아 스트림에 대한 추론을 수행합니다.
-   **입력 모드**:
    -   **Video Mode**: 동영상 파일 입력
    -   **RTSP Mode**: RTSP URL을 통한 네트워크 스트림 입력
    -   **Camera Mode**: 시스템에 연결된 카메라 입력
-   **미지원 예제**: Classification Task는 이미지 단위 분류에 특화되어 있어 `stream_inference` 기능을 제공하지 않습니다.
-   **성능 리포트**: 실행이 종료되면(비디오 종료, `Esc` 또는 `q`키 입력), 콘솔에 **전체 처리량(Overall FPS)** 등을 포함한 **`Performance Summary`** 를 출력합니다.
    -   **Avg Latency**: 각 파이프라인 단계의 평균 처리 시간 (밀리초)
    -   **Throughput**: 각 단계가 초당 처리할 수 있는 프레임 수 (FPS)
        -   **기본 계산 방식**: `1000ms / Avg Latency`
        -   **`async` 예제의 Inference Throughput**: NPU가 여러 프레임을 동시에 처리하므로 **Time Window** 방식으로 실제 처리량을 측정
            ```
            실제 처리량 = 처리한 프레임 수 / (마지막 완료 시각 - 첫 프레임 제출 시각)
            ```

    -   **추가 지표 (`async` 예제만 해당)**:
        -   `Infer Completed`: 완료된 총 Inference 수
        -   `Infer Inflight Avg`: `InferenceEngine`에 제출되어 있는 평균 프레임 수
            - 낮을수록: NPU에 충분한 작업이 공급되지 않음 (다른 단계가 병목)
            - 높을수록: NPU가 효과적으로 활용되고 있음
        -   `Infer Inflight Max`: `InferenceEngine`에 제출된 최대 프레임 수
    -   `--no-display` 옵션 사용 시, `Display` 항목은 리포트에서 제외됩니다.
    <details>
    <summary> Performance Summary 출력 예시 (sync 예제)</summary>

    ```
    ==================================================
                PERFORMANCE SUMMARY                
    ==================================================
    Pipeline Step   Avg Latency     Throughput     
    --------------------------------------------------
    Read                1.28 ms      783.1 FPS
    Preprocess          0.40 ms     2505.8 FPS
    Inference          28.27 ms       35.4 FPS
    Postprocess         5.89 ms      169.8 FPS
    Display            23.19 ms       43.1 FPS
    --------------------------------------------------
    Total Frames    :    300
    Total Time      :   17.7 s
    Overall FPS     :   16.9 FPS
    ==================================================
    ```
    
    **지표 설명**:
    - `sync` 예제는 모든 단계가 순차적으로 처리되므로, 각 단계의 `Throughput`이 `1000ms / Avg Latency`로 계산됩니다.
    - 이 예시에서는 `Overall FPS`(16.9)가 개별 단계들의 `Throughput`보다 훨씬 낮습니다.
    - 이는 모든 단계의 Latency가 순차적으로 누적되기 때문입니다: `1.28 + 0.40 + 28.27 + 5.89 + 23.19 = 59.03ms ≈ 16.9 FPS`
    </details>

    <details>
    <summary> Performance Summary 출력 예시 (async 예제)</summary>

    ```
    ==================================================
                PERFORMANCE SUMMARY                
    ==================================================
    Pipeline Step   Avg Latency     Throughput     
    --------------------------------------------------
    Read                1.95 ms      512.9 FPS
    Preprocess          1.10 ms      906.2 FPS
    Inference          67.29 ms      101.1 FPS*
    Postprocess         2.22 ms      450.9 FPS
    Display            16.54 ms       60.5 FPS
    --------------------------------------------------
    * Actual throughput via async inference
    --------------------------------------------------
    Infer Completed     :    300
    Infer Inflight Avg  :    5.9
    Infer Inflight Max  :      7
    --------------------------------------------------
    Total Frames        :    300
    Total Time          :    5.1 s
    Overall FPS         :   59.3 FPS
    ==================================================
    ```
    
    **지표 설명**:
    - **`Avg Latency` vs `Throughput` (Inference 단계)**:
        - `Avg Latency`(67.29ms): 개별 프레임이 `run_async()` 호출 시점부터 `wait()` 완료까지 걸린 평균 시간입니다. 이는 **`InferenceEngine` 내부 큐의 대기 시간**을 포함합니다.
        - `Throughput`(101.1 FPS*): Time Window 방식으로 측정한 NPU의 **실제 처리량**입니다. 큐 대기 시간과 무관하게 NPU가 초당 처리한 프레임 수를 나타냅니다.
    
    - **`Infer Inflight Avg`(5.9)의 의미**:
        - `InferenceEngine`에 제출되어 있는 평균 프레임 수입니다.
        - 예를 들어, DX-M1의 경우 NPU 코어는 3개이므로 최대 3개 프레임만 **동시에 처리**됩니다.
        - `Inflight Avg = 5.9`는 평균적으로 3개가 NPU에서 처리 중이고, 나머지 약 2.9개는 **`InferenceEngine` 내부 큐에서 대기** 중이었음을 의미합니다.
        - **`Inflight Avg`가 NPU 코어 수 이상이면** NPU가 쉬지 않고 효과적으로 활용되고 있으며, 큐에 충분한 작업이 대기 중인 이상적인 상태입니다.
        - 이 큐 대기 시간이 `Avg Latency`를 증가시키지만, NPU는 모든 코어를 활용하여 높은 처리량을 유지합니다.
    
    - 이 예시에서는 `Display` 단계(60.5 FPS)가 전체 파이프라인의 병목이 되어 `Overall FPS`가 59.3 FPS로 제한되지만, Inference 단계는 101.1 FPS의 처리량을 기록했습니다.
    </details>

---

## 6. 예제 실행하기

이 섹션에서는 `object_detection`의 `yolov9` 예제를 직접 실행하며 주요 기능과 Variant 별 처리 성능 차이를 확인합니다.

### 6.1. 기본 이미지 추론 (Sync)

가장 기본적인 `sync` 예제로 단일 이미지를 추론합니다. 아래 두 예제를 통해 후처리 구현 방식(Python/C++)에 따른 성능 차이를 비교해 보세요.

```bash
# Python 후처리
python src/python_example/object_detection/yolov9/yolov9_sync.py --model assets/models/YOLOV9S.dxnn --image sample/img/1.jpg

# C++ 후처리
python src/python_example/object_detection/yolov9/yolov9_sync_cpp_postprocess.py --model assets/models/YOLOV9S.dxnn --image sample/img/1.jpg
```

-   **실행 결과**
    - 객체 탐지 결과가 그려진 이미지가 화면에 나타납니다. 
    - 터미널에는 `Image Processing Summary`가 출력됩니다. 
    - 파이썬 후처리를 사용한 예제보다 C++ 후처리를 사용한 예제에서 `Postprocess` 단계의 Latency가 더 낮게 측정되는 것을 확인할 수 있습니다.
-   **💡 팁**: `--no-display` 옵션을 추가하면, 결과 이미지를 화면에 띄우는 대신 `artifacts/python_example/object_detection/` 폴더에 저장합니다.

### 6.2. 스트림 추론 및 성능 비교

이제 스트림 입력을 사용하여 `sync`와 `async`, 그리고 Python과 C++ 후처리의 성능을 비교해 봅니다.

#### 6.2.1. Sync vs. Async 성능 비교

동일한 비디오를 `sync`와 `async` 예제로 각각 실행하고 `Overall FPS`를 비교해 보세요.

```bash
# Sync 스트림 추론
python src/python_example/object_detection/yolov9/yolov9_sync.py --model assets/models/YOLOV9S.dxnn --video assets/videos/dance-group.mov

# Async 스트림 추론
python src/python_example/object_detection/yolov9/yolov9_async.py --model assets/models/YOLOV9S.dxnn --video assets/videos/dance-group.mov
```

-   **실행 결과**
    - 일반적으로 `sync` 예제보다 `async` 예제의 `Overall FPS`가  높게 측정됩니다. 이는 `async` 예제의 두 가지 최적화가 복합적으로 작용하기 때문입니다:
    1.  **파이프라인 병렬 처리**: `Preprocess`, `Inference`, `Postprocess` 등 파이프라인의 각 단계가 별도의 스레드에서 동시에 동작하여 CPU의 유휴 시간을 최소화합니다.
    2.  **NPU 처리량 개선**: `dx_engine`의 `InferenceEngine.run_async()` API를 통해 여러 추론 요청을 NPU에 미리 보내두고 병렬로 처리하므로, NPU 코어를 효과적으로 활용할 수 있습니다.

#### 6.2.2. Python PP vs. C++ PP 성능 비교

`async` 예제에 `--no-display` 옵션을 사용하여 순수 연산 성능 비교를 통해 후처리 최적화의 효과를 확인합니다.

```bash
# Async + Python 후처리
python src/python_example/object_detection/yolov9/yolov9_async.py --model assets/models/YOLOV9S.dxnn --video assets/videos/dance-group.mov --no-display

# Async + C++ 후처리
python src/python_example/object_detection/yolov9/yolov9_async_cpp_postprocess.py --model assets/models/YOLOV9S.dxnn --video assets/videos/dance-group.mov --no-display
```

-   **실행 결과**:
    -   `Postprocess` 단계의 처리량은 C++ 후처리가 일반적으로 더 높습니다.
    -   하지만 **`Overall FPS` 개선 여부는 파이프라인의 병목 위치에 따라 달라집니다**
    -   C++ 후처리는 다음 상황에서 `Overall FPS`를 향상시킬 수 있습니다:
    
        1. **후처리가 직접 병목인 경우**
            - `Postprocess` 단계의 `Throughput`이 다른 단계보다 현저히 낮을 때
            - C++ 후처리로 직접 가속
        
        2. **CPU 경합으로 인한 간접 병목인 경우** (특히 임베디드 환경)
            - Python 후처리의 높은 CPU 점유율이 `Read`, `Preprocess` 단계를 방해
            - `Infer Inflight Avg`가 낮게 측정되는 경우 (NPU에 데이터 공급 부족)
            - C++ 후처리로 CPU 경합을 해소하여 전체 파이프라인 성능 개선 가능

### 6.3. 다른 스트림 소스 사용하기

`--video` 인자 대신 `--rtsp` 또는 `--camera`를 사용하여 다른 스트림 소스에도 동일하게 예제를 적용할 수 있습니다.

```bash
# RTSP 입력
python src/python_example/object_detection/yolov9/yolov9_async.py --model assets/models/YOLOV9S.dxnn --rtsp rtsp://{YOUR_RTSP_URL}

# Camera 입력 (0번 카메라)
python src/python_example/object_detection/yolov9/yolov9_async.py --model assets/models/YOLOV9S.dxnn --camera 0
```

---

## 7. 고급: 모델별 성능 리포트 일괄 생성

6절의 튜토리얼에서는 `yolov9` 예제를 하나씩 실행하며 성능을 비교했습니다. 만약 **지원되는 모든 예제 Variants의 성능을 한 번에 측정하고, 그 결과를 표로 정리하여 비교**하고 싶다면 `pytest` 기반의 End-to-End(E2E) 테스트를 활용할 수 있습니다.

-   **핵심 기능**:
    -   `e2e` 테스트는 **실제 `.dxnn` 모델과 데이터(이미지, 비디오)를 사용**하여, 모든 예제 스크립트의 `image_inference`와 `stream_inference` 기능이 End-to-End로 정상 동작하는지 검증합니다.
    -   이 과정에서 **`stream_inference` 테스트의 Performance Summary 출력 결과를 수집**하여, 각 모델의 예제 Variant 별 성능이 정리된 `.csv` 리포트 파일을 `tests/python_example/performance_reports/` 폴더에 저장하고 요약된 E2E Performance Report를 콘솔에 출력합니다.


-   **실행 방법**:
    1.  먼저, 테스트에 필요한 의존성 패키지를 설치합니다.
        ```bash
        # (dx_app/ 경로에서 실행)
        pip install -r tests/python_example/requirements.txt
        ```
    2.  아래 명령어로 `e2e` 테스트를 실행합니다.
        ```bash
        # (dx_app/ 경로에서 실행)
        cd tests/python_example

        # 모든 e2e 테스트 실행
        pytest -m e2e
        
        # 또는 특정 모델(예: yolov9)의 e2e 테스트만 실행
        pytest -m "e2e and yolov9"
        ```
    
        <details>
        <summary> E2E Performance Report 출력 예시</summary>

        ```
        ===================================================================================================================================
         E2E Performance Report
        ===================================================================================================================================

         Object Detection - YOLOV9

         Model: assets/models/YOLOV9S.dxnn
         Video: assets/videos/dance-group.mov (478 frames)

         run_model FPS: (ORT ON) 101.7 FPS, (ORT OFF) 101.7 FPS

        -----------------------------------------------------------------------------------------------------------------------------------
        Variant                                    | E2E [FPS]    | Read [FPS]   | Preprocess [FPS]  | Inference [FPS]  | Postprocess [FPS]
        -----------------------------------------------------------------------------------------------------------------------------------
        yolov9_async                               | 100.5        | 606.8        | 603.8             | 101.6 *          | 197.6            
        yolov9_async_cpp_postprocess               | 100.4        | 673.2        | 624.2             | 101.5 *          | 670.6            
        yolov9_async_cpp_postprocess_ort_off       | 100.7        | 676.3        | 661.9             | 101.6 *          | 671.9            
        yolov9_async_ort_off                       | 100.7        | 619.8        | 748.8             | 101.5 *          | 179.4            
        yolov9_sync                                | 25.5         | 327.9        | 999.7             | 35.0             | 151.4            
        yolov9_sync_cpp_postprocess                | 29.8         | 310.7        | 904.4             | 35.8             | 740.0            
        yolov9_sync_cpp_postprocess_ort_off        | 30.1         | 310.0        | 878.5             | 36.4             | 699.2            
        yolov9_sync_ort_off                        | 24.6         | 357.5        | 1111.1            | 35.5             | 113.9            
        -----------------------------------------------------------------------------------------------------------------------------------
        ```
        </details>

-   **결과 활용**:
    -   생성된 `.csv` 리포트를 통해 각 모델마다 현재의 컴퓨팅 환경에서 어떤 처리 방식 조합이 가장 높은 성능을 보이는지 한눈에 비교하고, 자신의 애플리케이션에 적용할 최적화 전략을 수립할 수 있습니다.
    -   테스트 프레임워크에 대한 더 자세한 내용은 [`tests/python_example/README.md`](../../../tests/python_example/README.md)를 참고하십시오.
