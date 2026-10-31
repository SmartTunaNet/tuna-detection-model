# 정치망 참치 탐지·카운트 모델

영덕 정치망 입구의 수중 카메라 영상에서 참치를 찾아 세고, 쿼터에 닿으면 입구 차단 신호를 내는 모델입니다. 지금은 공개 데이터만으로 만든 개념 검증 단계입니다.

- 모델: YOLO11s, 클래스 2개 (`0 = tuna`, `1 = other_fish`)
- 추적: ByteTrack
- 우선순위: 재현율 (참치를 놓치지 않는 것)
- 용도: 비상업 연구

## 현재 성능

학습에 쓰지 않은 탁한 물·어군 사진 43장(참치 501마리)으로 잰 값이 대표 수치입니다.

| 검증 데이터 | 참치 재현율 | 참치 정밀도 |
|---|---|---|
| 탁한 물·어군 43장 (대표) | 37.3% | 39.0% |
| 맑은 물 61장 | 94.8% | 93.5% |
| 두 검증 합산 | 42.6% | 45.7% |

- 맑은 물 검증에는 학습에 쓴 영상과 같은 영상의 프레임이 섞여 있어 실제보다 높게 나옵니다.
- 탁한 물·어군에서 물고기 자체는 54.9%를 찾습니다. 재현율이 낮은 주원인은 참치를 다른 물고기로 분류하는 것입니다.
- 처리 속도는 M3 Pro 노트북에서 프레임당 약 13~15ms(초당 약 70프레임)입니다.
- 현장(영덕) 영상은 아직 없습니다.

자세한 설명과 예상 질문은 [report/expected_questions.md](report/expected_questions.md)에 있습니다.

## 설치

Python 3.12에서 확인했습니다.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

라벨 수정에 쓰는 Label Studio는 패키지 충돌을 피하려고 별도 가상환경에 설치합니다.

```bash
python3.12 -m venv .venv-ls
.venv-ls/bin/pip install label-studio
```

## 모델 가중치

가중치 파일은 저장소에 넣지 않고 [Releases](https://github.com/SmartTunaNet/tuna-detection-model/releases)에 올려 두었습니다. `best.pt`를 받아 `weights/` 폴더에 넣으세요.

```bash
mkdir -p weights
gh release download v3 -R SmartTunaNet/tuna-detection-model -p best.pt -D weights
```

## 바로 써 보기: 영상에서 참치 세기

```bash
.venv/bin/python scripts/count_video.py weights/best.pt <입력영상.mp4> <출력영상.mp4> --quota 20
```

- 화면 왼쪽 위에 "참치 N / 쿼터 Q"와 "입구 개방 / 입구 차단"이 표시됩니다.
- 박스 색: 빨강 = 참치, 하늘색 = 기타
- 한 물고기가 8프레임 이상 보이고, 그동안의 신뢰도(confidence) 합이 참치 쪽이 더 크면 참치 한 마리로 셉니다.
- `--line x1,y1,x2,y2`: 화면 비율(0~1)로 선을 그어, 선을 넘어간 참치만 셉니다. 만들어 두었지만 아직 시험하지 않았습니다.
- `--show`: 처리하면서 창으로 결과를 보여 줍니다. `q` 또는 `Esc`로 멈춥니다.
- 출력 영상 옆에 `.events.csv`(센 시각, 입구 차단 시각)와 `.summary.txt`가 생깁니다.
- Apple Silicon(MPS) 기준으로 작성했습니다. 다른 환경에서는 스크립트의 `device="mps"`를 바꿔야 합니다.

## 전체 흐름

데이터 폴더(`data/`)는 저작권이 있는 사진과 영상이 섞여 있어 저장소에 넣지 않았습니다. 아래 순서로 다시 만들 수 있습니다.

| 단계 | 스크립트 | 하는 일 |
|---|---|---|
| 1. 수집 | `download_roboflow.sh`, `download_inat.py` | Roboflow 데이터셋과 iNaturalist CC 라이선스 사진을 받습니다. Roboflow는 `ROBOFLOW_API_KEY` 환경 변수가 필요합니다. |
| 2. 수중 사진 고르기 | `filter_underwater.py` | CLIP으로 수중 사진일 확률을 매깁니다. 갑판, 어시장, 그림 사진을 거르는 용도입니다. |
| 3. 자동 라벨 | `autolabel.py` | Grounding DINO로 물고기 박스를 그리고, CLIP으로 잠수부·물거품 같은 박스를 버린 뒤 참치/기타를 나눕니다. |
| 4. 사람 검수 | `review_sheets.py`, `contact_sheet.py` | 사진을 격자로 모아 합격·불합격을 판정하기 쉽게 만듭니다. |
| 5. 라벨 수정 | `prelabel.py` → `ls_import.py` → `ls_export.py` | 모델이 박스를 미리 그리고, Label Studio에서 사람이 고친 뒤 YOLO 라벨로 내보냅니다. |
| 6. 데이터셋 구성 | `build_dataset.py` | 학습 / 맑은 물 검증 / 어려운 검증으로 나눕니다. 어려운 검증은 영상 단위로 통째로 떼어 학습과 겹치지 않게 합니다. |
| 7. 학습 | `train.py` | YOLO11s를 학습합니다. 물빛 변화를 흉내 내려고 색 증강을 기본값보다 강하게 겁니다. 시작 가중치를 `weights/` 폴더에서 읽으므로, 처음 학습할 때는 Ultralytics의 `yolo11s.pt`를 `weights/`에 넣어 두세요. |
| 8. 평가 | `eval_recall.py` | 판정 기준별 참치 재현율, 종과 무관한 물고기 검출률, 사진당 오탐 박스 수를 셉니다. |
| 9. 영상 카운트 | `count_video.py` | 위 "바로 써 보기" 참고 |

각 스크립트의 정확한 사용법은 파일 맨 위 설명에 있습니다.

### Label Studio로 라벨 고치기

```bash
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$(pwd)/data \
.venv-ls/bin/label-studio start --no-browser -p 8080
```

- 브라우저에서 http://localhost:8080 에 가입한 뒤, 계정 설정에서 Personal Access Token을 만들어 `LS_PAT` 환경 변수로 넘깁니다.
- 토큰은 코드나 저장소에 적지 마세요.
- `label-studio --version`처럼 옵션만 붙여 실행해도 서버가 켜질 수 있으니 주의하세요.

```bash
.venv/bin/python scripts/prelabel.py weights/best.pt data/relabel
LS_PAT=<토큰> .venv/bin/python scripts/ls_import.py data/relabel "tuna relabel"
# Label Studio에서 수정 후
LS_PAT=<토큰> .venv/bin/python scripts/ls_export.py <프로젝트 번호> data/relabel
```

## 알려진 한계

- 현장(영덕 정치망) 영상이 없습니다. 국내 참다랑어 양식장 수중 영상으로 시험했을 때 참치를 대부분 '기타'로 분류했습니다.
- 빽빽한 참치 떼에서는 박스를 거의 못 그리거나 '기타'로 분류합니다.
- 참치가 없는 고등어 떼 영상(56초)에서 65마리를 참치로 셌습니다.
- '다른 물고기' 학습 사진 1,129장은 자동 라벨만 붙였고 사람이 검수하지 않았습니다.
- 카운트 정확도(실제 마릿수 대비)는 측정하지 않았습니다.
- 입구 차단 장치와는 연결되어 있지 않습니다. 신호는 화면과 기록 파일에만 남습니다.

## 데이터 출처와 라이선스

- [brin Tuna Fish Detection](https://universe.roboflow.com/brin-computer-vision-research-team/tuna-fish-detection-xbhqv/dataset/15) (Roboflow Universe, CC BY 4.0)
- red_tuna (Roboflow Universe)
- iNaturalist 사진 (CC BY / CC BY-NC, 사진별 라이선스와 촬영자는 `download_inat.py`가 `meta.csv`에 기록)
- 유튜브 공개 영상 (데모·평가용. CC 라이선스가 아니므로 공개·상업적 사용 전 원 촬영자의 허락이 필요합니다)

비상업 연구 목적으로만 사용합니다.
