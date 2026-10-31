# 사람이 만든 라벨 백업

사진은 저작권 때문에 저장소에 넣지 않고, 사람이 직접 만든 판단만 보관합니다.

- `relabel_fixed/`: Label Studio에서 직접 고친 사진 153장의 YOLO 라벨 (`0 = tuna`, `1 = other_fish`). 파일 이름은 `<출처>__<원본 파일 이름>`.
- `excluded.txt`: Label Studio에서 "exclude"로 뺀 사진 목록.
- `order.txt`: Label Studio에 올린 순서 (수중 사진일 확률 높은 순).
- `review/`: 자동 라벨 검수에서 합격한 사진 목록 (`<출처>_accept.txt`).
- `splits/`: 최종 데이터셋(tuna_v3)의 학습 / 맑은 물 검증 / 어려운 시험(val_hard) 파일 목록.

원본 사진을 다시 받은 뒤 이 파일들을 `data/relabel/fixed/`, `data/review/`에 넣고 `scripts/build_dataset.py`를 돌리면 같은 데이터셋을 다시 만들 수 있습니다.
