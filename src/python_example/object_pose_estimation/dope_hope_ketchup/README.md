# DOPE Hope-Ketchup — 6DoF Object Pose Estimation 예제 (데모)

belief map 기반으로 cuboid의 8 vertex + centroid를 검출하고, PnP로 6DoF pose를
추정해 3D cuboid를 영상에 overlay하는 DOPE 예제입니다. Python 4 variant와
C++ 2 variant를 제공합니다.

| 항목 | 값 |
|------|-----|
| Model | `dope_hope_ketchup_1` (`dope-hope-ketchup-1.dxnn`) |
| Task type | `object_pose_estimation` |
| Input | UINT8 image `[1, 480, 640, 3]` |
| Input 해상도 | 640 × 480 |
| Output | `[1, 25, 60, 80]` — ch 0–8 belief maps(8 vertex + centroid), ch 9–24 affinity fields |
| Object | Hope-Ketchup 단일 객체 |

---

## ⚠️ 데모 한정 — 알고리즘 범위와 한계

이 예제는 **시각화 데모**입니다. 원본 [NVlabs/Deep_Object_Pose](https://github.com/NVlabs/Deep_Object_Pose)
의 전체 후처리 대신 `dx-modelzoo`의 단순화된 `dope_decode`를 따릅니다.

| 항목 | 이 예제(데모) | NVlabs 원본 |
|------|---------------|-------------|
| Peak 검출 | belief map 채널별 **argmax 1개** | gaussian blur(σ=3) 후 local-maxima **다중 peak** |
| Affinity fields (ch 9–24) | **미사용** | vertex→centroid vector voting (association) |
| 다중 객체 | **불가 (단일 인스턴스 전용)** | 다중 객체 association 지원 |
| 검출 게이팅 | `conf_threshold`(centroid 신뢰도) | `thresh_map` / `thresh_points` / `thresh_angle` |
| Camera intrinsic | **추정값** (focal = image width) | 실제 calibration(`camera_info.yaml`) |
| Pose 정확도 | overlay는 그럴듯하나 **metric 정확도 보장 X** | calibration 시 metric 정확 |

> **결론**: 화면에 객체가 1개일 때의 시각적 데모로는 충분합니다. 다중 객체나
> metric pose가 필요하면 affinity 기반 association과 실제 camera calibration이
> 추가로 필요합니다.

---

## 1. 사전 준비

```bash
./install.sh && ./build.sh                 # C++ 예제 + dx_postprocess pybind11 빌드
./setup.sh --models dope_hope_ketchup_1     # 모델 다운로드
dxrt-cli -s                                 # NPU 확인
```

> `*_cpp_postprocess.py` variant는 `dx_postprocess` pybind11 모듈을 사용합니다.
> `src/postprocess/dope/`를 수정하면 모듈을 **재빌드**해야 반영됩니다.

---

## 2. 파일 구조

```
src/python_example/object_pose_estimation/dope_hope_ketchup/
├── README.md                                       # (이 문서)
├── config.json                                     # 튜닝 파라미터 (아래 §5)
├── factory/dope_hope_ketchup_factory.py            # IPoseFactory 구현
├── dope_hope_ketchup_sync.py                       # ① Python sync
├── dope_hope_ketchup_async.py                      # ② Python async
├── dope_hope_ketchup_sync_cpp_postprocess.py       # ③ Python sync + C++ postprocess
└── dope_hope_ketchup_async_cpp_postprocess.py      # ④ Python async + C++ postprocess

# 공용 컴포넌트
src/python_example/common/processors/dope_postprocessor.py   # peak 검출 + PnP(pose)
src/python_example/common/processors/_dope_geometry.py       # 3D cuboid / camera / PnP 공용 헬퍼
src/python_example/common/visualizers/dope_visualizer.py     # cuboid overlay

# C++
src/postprocess/dope/                                        # C++ peak 후처리
src/cpp_example/object_pose_estimation/dope_hope_ketchup/    # C++ sync/async + factory(PnP+overlay)
```

---

## 3. 실행 방법

### 3-1. Python (`src/python_example/` 에서 실행)

```bash
cd src/python_example/object_pose_estimation/dope_hope_ketchup

# ① sync — 단일 이미지
python dope_hope_ketchup_sync.py \
  -m ../../../../assets/models/dope-hope-ketchup-1.dxnn \
  -i <image.jpg>

# ② async
python dope_hope_ketchup_async.py  -m <model.dxnn> -i <image 또는 디렉터리>

# ③ / ④ C++ postprocess variant (dx_postprocess 모듈 필요)
python dope_hope_ketchup_sync_cpp_postprocess.py   -m <model.dxnn> -i <image>
python dope_hope_ketchup_async_cpp_postprocess.py  -m <model.dxnn> -i <디렉터리>
```

입력을 생략하면 task 기본 샘플 이미지를 사용합니다. `--config`를 생략하면 이 폴더의
`config.json`이 자동 적용됩니다. `-h`/`--help`로 전체 옵션을 확인하세요.

### 3-2. C++ (빌드 산출 바이너리)

> **빌드 선행 필요**: DOPE C++ 예제는 기본 빌드에서 제외돼 있습니다
> (`solvePnP`/`projectPoints`용 `opencv_calib3d` 의존성). 빌드하려면
> `src/cpp_example/CMakeLists.txt`의 `CATEGORIES`에서 `object_pose_estimation`
> 주석을 해제하고 calib3d를 확보한 뒤 `./build.sh`를 실행하세요.
> 바이너리는 `dope_hope_ketchup_sync` / `dope_hope_ketchup_async` 로 생성됩니다.

```bash
# ⑤ sync — 단일 이미지
./dope_hope_ketchup_sync \
  -m assets/models/dope-hope-ketchup-1.dxnn \
  -i <image.jpg>

# ⑥ async — 이미지 디렉터리
./dope_hope_ketchup_async \
  -m assets/models/dope-hope-ketchup-1.dxnn \
  -i <image_dir>

# 비디오 / 카메라 / RTSP 입력
./dope_hope_ketchup_sync -m <model.dxnn> -v <video.mp4>
./dope_hope_ketchup_sync -m <model.dxnn> -c 0
./dope_hope_ketchup_sync -m <model.dxnn> -r <rtsp://...>

# 저장 + headless (창 출력 없이 결과만 저장)
./dope_hope_ketchup_async -m <model.dxnn> -i <image_dir> --save --no-display
```

| 옵션 | 설명 |
|------|------|
| `-m`, `--model_path` | `.dxnn` 모델 경로 (필수) |
| `-i`, `--image_path` | 이미지 파일 **또는** 디렉터리 |
| `-v`, `--video_path` | 비디오 파일 경로 |
| `-c`, `--camera_index` | 카메라 장치 index |
| `-r`, `--rtsp_url` | RTSP 스트림 URL |
| `-s`, `--save` | 결과를 디스크에 저장 |
| `--save-dir` | 저장 베이스 디렉터리 (기본 `artifacts/cpp_example`) |
| `--no-display` | 창 출력 비활성화 (headless) |
| `--config` | `config.json` 경로 |
| `-l`, `--loop` | 추론 반복 횟수 |
| `--show-log` | 상세 로그 출력 (기본은 조용함) |
| `-h`, `--help` | 옵션 도움말 |

> 입력 소스(`-i`/`-v`/`-c`/`-r`)는 **정확히 1개**만 지정해야 합니다.
> 입력을 모두 생략하면 task 기본 샘플 이미지를 사용합니다.

---

## 4. 출력 (`DopeResult`)

Python postprocessor(`DOPEPostprocessor`)가 반환하는 detection 1건:

| 필드 | 내용 |
|------|------|
| `keypoints` | `(9, 2)` 정규화 `[0,1]` 좌표 (8 vertex + centroid) |
| `centroid` | `(2,)` centroid 정규화 좌표 |
| `confidence` | centroid belief peak 값 |
| `all_conf` | `(9,)` 채널별 peak 값 |
| `pose` | `{'R': (3,3), 't': (3,), 'rvec': (3,1), 'tvec': (3,1)}` 또는 `None` |
| `image_width` / `image_height` | 원본 이미지 크기 (px 환산용) |

> `pose`는 `solve_pnp=true`이고 vertex가 4개 이상 유효할 때 채워집니다. visualizer는
> 이 `pose`를 그대로 재사용해 cuboid를 투영하므로, 그려지는 박스와 `pose`가 항상 일치합니다.

---

## 5. `config.json` — 튜닝 파라미터

```json
{
    "conf_threshold": 0.0,
    "subpixel_refine": true,
    "solve_pnp": true,
    "object_size_cm": [14.860799789428711, 4.3368000984191895, 6.4513998031616211],
    "focal_length": null
}
```

| 키 | 기본값 | 설명 |
|----|--------|------|
| `conf_threshold` | `0.0` | centroid belief peak이 이 값보다 낮으면 검출을 버립니다. **0.0은 "항상 검출"(데모 기본)**. 객체가 없을 때 헛검출을 줄이려면 `0.1~0.3` 권장 (affinity/thresh_map이 없으므로 이 값이 유일한 게이트). |
| `subpixel_refine` | `true` | argmax 주변 3×3 belief-weighted centroid로 sub-pixel 보정. `false`면 정수 peak 사용. |
| `solve_pnp` | `true` | `false`면 PnP를 풀지 않고 `pose=None` (keypoint만 필요할 때). |
| `object_size_cm` | Ketchup `[14.86, 4.34, 6.45]` | 대상 객체의 실제 cuboid 크기 `[width, height, depth]`(cm). **다른 객체면 반드시 교체**해야 pose가 맞습니다. |
| `focal_length` | `null` | 카메라 focal length(px). `null`이면 데모 추정값(= image width)을 사용. **metric pose가 필요하면 실제 calibration 값을 넣으세요** (§6). |

> postprocessor와 visualizer는 `object_size_cm`·`focal_length`를 **동일하게** 사용하도록
> factory가 양쪽에 주입합니다. 한쪽만 바꾸면 overlay와 `pose`가 어긋납니다.

---

## 6. Camera intrinsic — 실제 calibration (중요)

기본 camera matrix는 **추정값**입니다:

```
K = [[ f, 0, W/2 ],
     [ 0, f, H/2 ],   f = focal_length (기본: image width W)
     [ 0, 0,  1  ]]   principal point = 이미지 중심, distortion = 0
```

- 이 상태에서도 cuboid overlay는 그럴듯하게 보이지만, `pose`의 **translation(`t`)·
  rotation(`R`)은 metric 정확도가 없습니다**.
- 실제 6DoF pose가 필요하면:
  1. 카메라를 calibration해 focal length(px)와 principal point, distortion을 구하고,
  2. `config.json`의 `focal_length`에 focal(px)을 설정하세요.
  3. principal point가 이미지 중심이 아니거나 distortion이 큰 카메라라면
     `_dope_geometry.build_camera_matrix()` / `solve_pose()`의 `K`·`dist`를
     실제 calibration 값으로 교체해야 합니다 (현재 데모는 중심·무왜곡 가정).

---

## 7. 시각화

- 8 vertex(컬러 점) + centroid(흰 점) 표시
- PnP 성공 시 12개 cuboid edge(녹색) + 윗면 X(노란색) overlay
- 유효 vertex가 4개 미만이면 fallback으로 검출된 peak를 직접 연결

---

## 8. 트러블슈팅

| 증상 | 원인 / 해결 |
|------|-------------|
| 객체가 없는데도 박스가 그려짐 | affinity/`thresh_map` 게이팅이 없는 데모 특성. `config.json`의 `conf_threshold`를 올리세요(예: `0.2`). |
| 객체가 2개 이상인데 박스가 깨짐 | 단일 인스턴스 전용(채널별 argmax 1개). 다중 객체는 affinity association 구현 필요 — 데모 범위 밖. |
| cuboid 크기/형상이 안 맞음 | `object_size_cm`가 대상 객체와 다름 → 실제 치수로 교체. |
| pose 값(거리)이 비현실적 | `focal_length`가 추정값(image width). 실제 calibration focal을 설정(§6). |
| `pose`가 항상 `None` | `solve_pnp:false`이거나 유효 vertex < 4. config 확인. |
| C++ postprocess 결과가 다름 | `src/postprocess/dope/` 수정 후 `dx_postprocess` 미재빌드 → `./build.sh` 재실행. |
