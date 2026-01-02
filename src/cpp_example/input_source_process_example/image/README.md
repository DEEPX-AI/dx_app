# Image Processing Examples

이 디렉토리에는 DXRT를 사용한 이미지 처리 예제들이 있습니다.

## 빌드 방법

### 기본 예제 (CMake 사용)

OpenCV를 사용하는 기본 예제들은 CMake를 사용하여 빌드할 수 있습니다:

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

빌드되는 실행 파일:
- `image_opencv` - OpenCV를 사용한 이미지 로더 예제
- `image_multi_model_test` - 멀티 모델 테스트 예제

### image_libjpeg.cpp (별도 빌드)

`image_libjpeg.cpp`는 libjpeg-turbo 라이브러리가 필요한 예제로, 별도로 빌드해야 합니다.

#### 1. 필수 패키지 설치

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install libturbojpeg-dev libjpeg-turbo8-dev
```

**CentOS/RHEL/Rocky Linux:**
```bash
sudo yum install turbojpeg-devel libjpeg-turbo-devel
# 또는 dnf를 사용하는 경우
sudo dnf install turbojpeg-devel libjpeg-turbo-devel
```

**Fedora:**
```bash
sudo dnf install turbojpeg-devel libjpeg-turbo-devel
```

#### 2. 수동 컴파일

```bash
# 프로젝트 루트 디렉토리에서
cd src/cpp_example/input_source_process_example/image

# C++17 사용 (권장)
g++ -std=c++17 -O3  -Wall -Wextra -Wpedantic \
    -I../../../../extern/ \
    -I../../../../src/utility/ \
    -DPROJECT_ROOT_DIR="$(pwd)/../../../../" \
    image_libjpeg.cpp \
    ../../../../src/utility/common_util.cpp \
    -ldxrt -lturbojpeg -pthread -lstdc++fs \
    -o ../../../../bin/image_libjpeg

# 또는 C++11 사용 (filesystem 라이브러리가 experimental인 경우)
g++ -std=c++11 -O3  -Wall -Wextra -Wpedantic \
    -I../../../../extern/ \
    -I../../../../src/utility/ \
    -DPROJECT_ROOT_DIR="$(pwd)/../../../../" \
    image_libjpeg.cpp \
    ../../../../src/utility/common_util.cpp \
    -ldxrt -lturbojpeg -pthread -lstdc++fs \
    -o ../../../../bin/image_libjpeg
```

#### 3. pkg-config를 사용한 컴파일 (권장)

시스템에 pkg-config가 설치되어 있다면 더 간단하게 컴파일할 수 있습니다:

```bash
# TurboJPEG 라이브러리 정보 확인
pkg-config --exists libturbojpeg && echo "TurboJPEG found" || echo "TurboJPEG not found"

# 컴파일
g++ -std=c++17 -O3  -Wall -Wextra -Wpedantic \
    -I../../../../extern/ \
    -I../../../../src/utility/ \
    -DPROJECT_ROOT_DIR="$(pwd)/../../../../" \
    image_libjpeg.cpp \
    ../../../../src/utility/common_util.cpp \
    $(pkg-config --cflags --libs libturbojpeg) \
    -ldxrt -pthread -lstdc++fs \
    -o ../../../../bin/image_libjpeg
```

## 사용법

### image_opencv
```bash
./bin/image_opencv -m path/to/model.dxnn -i path/to/image.jpg
```

### image_multi_model_test
```bash
./bin/image_multi_model_test -m path/to/model.dxnn -i path/to/image.jpg
```

### image_libjpeg
```bash
./bin/image_libjpeg -m path/to/model.dxnn -i path/to/image.jpg --width 224 --height 224
```

## 특징

### image_libjpeg.cpp의 장점

1. **OpenCV 독립적**: OpenCV 없이도 JPEG 이미지를 처리할 수 있습니다.
2. **고성능**: libjpeg-turbo는 SIMD 최적화를 통해 빠른 JPEG 디코딩을 제공합니다.
3. **메모리 효율적**: 필요한 크기로 직접 리사이즈하여 메모리 사용량을 최소화합니다.
4. **비동기 처리**: DXRT의 비동기 추론 기능을 활용합니다.

### 의존성 비교

| 예제 | OpenCV | libjpeg-turbo | 기타 |
|------|--------|---------------|------|
| image_opencv | ✅ | ❌ | - |
| image_multi_model_test | ✅ | ❌ | - |
| image_libjpeg | ❌ | ✅ | cxxopts, filesystem |

## 문제 해결

### 라이브러리 찾기 오류

만약 컴파일 시 라이브러리를 찾을 수 없다는 오류가 발생하면:

1. **TurboJPEG 설치 확인**:
   ```bash
   ldconfig -p | grep turbojpeg
   ```

2. **헤더 파일 위치 확인**:
   ```bash
   find /usr -name "turbojpeg.h" 2>/dev/null
   ```

3. **라이브러리 경로 추가** (필요한 경우):
   ```bash
   export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   ```

### filesystem 라이브러리 링크 오류

C++17 filesystem이 지원되지 않는 구형 컴파일러에서는 `-lstdc++fs` 대신 다른 옵션을 사용해야 할 수 있습니다:

- GCC < 9.0: `-lstdc++fs`
- Clang < 9.0: `-lc++fs`
- MSVC: 별도 링크 불필요

## 라이선스

이 예제들은 프로젝트의 라이선스를 따릅니다.