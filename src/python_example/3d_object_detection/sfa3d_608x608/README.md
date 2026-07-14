# SFA3D 608x608 — LiDAR 3D Object Detection 예제

KITTI LiDAR point cloud를 BEV(Bird's-Eye-View)로 변환해 3D object를 검출하는 SFA3D
예제입니다. Python 4 variant와 C++ 2 variant를 제공합니다.

| 항목 | 값 |
|------|-----|
| Model | `sfa3d_608x608` (`sfa3d_608x608.dxnn`) |
| Task type | `3d_detection` |
| Input | LiDAR point cloud (`.bin`) — KITTI velodyne 포맷 |
| Input 해상도 | 608 × 608 (BEV) |
| Class | `Pedestrian`, `Car`, `Cyclist` |

---

## 1. 사전 준비

### 1-1. 빌드

```bash
./install.sh && ./build.sh      # C++ 예제 + dx_postprocess pybind11 빌드 (Linux)
# Windows: build.bat --minimal
```

- C++ 예제(`sfa3d_608x608_sync` / `_async`)는 위 빌드로 생성됩니다.
- Python `*_cpp_postprocess.py` variant는 `dx_postprocess` pybind11 모듈이 필요합니다(빌드에 포함).
  > **중요**: SFA3D 후처리(`src/postprocess/sfa3d/`)를 수정하면 `dx_postprocess.so`를 **반드시 재빌드**해야 합니다.
  > 구버전 모듈 사용 시 `z3d`/`dim` 값이 틀려 Cam+Box 시각화가 깨집니다.

### 1-2. 모델 다운로드

```bash
./setup.sh --models sfa3d_608x608     # → assets/models/sfa3d_608x608.dxnn
```

### 1-3. NPU 확인

```bash
dxrt-cli -s
```

---

## 2. 파일 구조

```
src/python_example/3d_object_detection/sfa3d_608x608/
├── README.md                                       # (이 문서)
├── config.json                                     # score/nms threshold 등
├── calib_policy.py                                 # calib 동반 파일 정책
├── factory/sfa3d_608x608_factory.py         # IDetectionFactory 구현
├── sfa3d_608x608_sync.py                    # ① Python sync
├── sfa3d_608x608_async.py                   # ② Python async
├── sfa3d_608x608_sync_cpp_postprocess.py    # ③ Python sync + C++ postprocess
└── sfa3d_608x608_async_cpp_postprocess.py   # ④ Python async + C++ postprocess

src/cpp_example/3d_object_detection/sfa3d_608x608/
├── config.json
├── factory/sfa3d_608x608_factory.hpp        # I3DDetectionFactory 구현
├── sfa3d_608x608_sync.cpp                   # ⑤ C++ sync
└── sfa3d_608x608_async.cpp                  # ⑥ C++ async
```

---

## 3. 실행 방법

### 3-1. Python (`src/python_example/` 에서 실행)

```bash
cd src/python_example/3d_object_detection/sfa3d_608x608

# ① sync — 단일 .bin
python sfa3d_608x608_sync.py \
  -m ../../../../assets/models/sfa3d_608x608.dxnn \
  -i ../../../../sample/kitti/velodyne/000049.bin

# ② async — velodyne 디렉터리 batch
python sfa3d_608x608_async.py \
  -m ../../../../assets/models/sfa3d_608x608.dxnn \
  -i ../../../../sample/kitti/velodyne

# ③ / ④ C++ postprocess variant (dx_postprocess 모듈 필요, 미설치 시 Python postprocessor로 fallback)
python sfa3d_608x608_sync_cpp_postprocess.py  -m <model.dxnn> -i <bin 또는 velodyne 디렉터리>
python sfa3d_608x608_async_cpp_postprocess.py -m <model.dxnn> -i <velodyne 디렉터리>
```

입력을 생략하면 기본 샘플 `sample/kitti/velodyne/000049.bin` 을 사용합니다.

### 3-2. C++ (빌드 산출 바이너리)

```bash
# sync — 단일 .bin
./sfa3d_608x608_sync \
  -m assets/models/sfa3d_608x608.dxnn \
  -i sample/kitti/velodyne/000049.bin

# async — velodyne 디렉터리 batch + 저장 + headless
./sfa3d_608x608_async \
  -m assets/models/sfa3d_608x608.dxnn \
  -i sample/kitti/velodyne \
  --calib-dir sample/kitti/calib \
  --image2-dir sample/kitti/image_2 \
  --save --no-display
```

---

## 4. CLI 옵션

| 옵션 | 설명 |
|------|------|
| `-m`, `--model` | `.dxnn` 모델 경로 (필수) |
| `-i`, `--image` | LiDAR `.bin` 단일 파일 **또는** velodyne 디렉터리 |
| `--calib-dir` | `{frame_id}.txt` calib 파일 디렉터리 (`-i` stem과 짝) |
| `--image2-dir` | `{frame_id}.png/.jpg` 카메라 이미지 디렉터리 (`-i` stem과 짝) |
| `--save`, `-s` | 결과 이미지 저장 |
| `--no-display` | 창 출력 비활성화 (headless) |
| `--config` | `config.json` 경로 (기본 자동 탐지) |
| `--loop`, `-l` | 추론 반복 횟수 |
| `--show-log` | 프레임별 상세 로그 출력 (기본은 조용함) |

> 기본 KITTI 레이아웃(`velodyne/` + `calib/` + `image_2/`)이면 `--calib-dir`/`--image2-dir`는 생략 가능합니다.
> Python·C++ 모두 `-h`/`--help`로 옵션을 확인할 수 있습니다.

---

## 5. 입력 / 샘플

- **지원 입력**: `--image` (`.bin` 단일 파일 또는 velodyne 디렉터리)
- **미지원 입력**: `--video`, `--camera`, `--rtsp` (LiDAR 전용 파이프라인)

| Frame | 설명 |
|-------|------|
| `000049` | Car — 기본 데모 |
| `000535` | Car / Pedestrian |

샘플 데이터: `sample/kitti/{velodyne,calib,image_2,label_2}/`

---

## 6. `config.json`

```json
{
    "score_threshold": 0.3,
    "nms_threshold": 0.2,
    "max_detections": 100,
    "require_calib": false
}
```

| 키 | 설명 |
|----|------|
| `score_threshold` | 검출 confidence 임계값 |
| `nms_threshold` | NMS IoU 임계값 |
| `max_detections` | 최대 검출 개수 |
| `require_calib` | `true`면 calib 파일이 없을 때 에러 (기본 `false`) |

---

## 7. 시각화 (BEV)

- **모델 입력 BEV**는 변경하지 않습니다 (preprocessor grid: +y → 오른쪽 열).
- **표시용**으로만 BEV raster를 먼저 수평 flip한 뒤, box/legend/label을 display 좌표 `col = (y_max - y)`로 그립니다.
- 패널 제목(`"BEV"` 등)은 flip 이후에 그려져 글자가 뒤집히지 않습니다.
- calib(`--calib-dir`)와 카메라 이미지(`--image2-dir`)가 있으면 Cam+Box(원근 투영) 뷰가 함께 렌더링됩니다.

---

## 8. `--show-log` 동작

다른 dx_app 예제와 동일합니다.

| 출력 | `--show-log` 없음 | `--show-log` 있음 |
|------|-------------------|-------------------|
| Model loaded / input size | O | O |
| Starting inference | O | O |
| PERFORMANCE SUMMARY | O | O |
| Config loaded | X | O |
| Input 경로 / 해상도 | X | O |
| `[Result] Detected N...` / 검출 좌표 | X | O |
| 프레임별 Read/Pre/Infer ms | X | O |

---

## 9. 트러블슈팅

| 증상 | 원인 / 해결 |
|------|-------------|
| `dx_postprocess.SFA3DPostProcess not available` | pybind 모듈 미설치 → Python postprocessor로 자동 fallback. C++ postprocess를 쓰려면 `./build.sh`로 `dx_postprocess` 재빌드 |
| Cam+Box 뷰가 깨짐 | 구버전 `dx_postprocess.so` 사용 → 후처리 수정 후 모듈 재빌드 |
| 모델을 찾지 못함 | `./setup.sh --models sfa3d_608x608` 로 다운로드 후 `-m` 경로 확인 |
| `--video`/`--camera` 동작 안 함 | SFA3D는 LiDAR `.bin` 입력 전용 (의도된 제한) |
