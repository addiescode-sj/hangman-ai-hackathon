# TASK.md — 선박 매칭 로드맵

상태: `[ ]` 대기 · `[~]` 진행 · `[x]` 완료. 각 task는 **/plan-first <ID>** 로 시작한다 (계획 → 옵션 3개 → 검증 → 구현).
Task 완료 시 산출물 경로와 측정 수치를 한 줄로 덧붙인다.

## P0. 기준선 · 평가 체계
- [skip] T0.1 baseline_submit 실행 재현 (환경, 실행 시간, 출력 형식 확인)
- [x] T0.2 holdout split 고정 (seed, 비율, train target 중복 처리 규칙) → `src/split.py` — StratifiedGroupKFold(5, seed 42, 그룹=target, 층=한글) fold 0, holdout 2,510행/한글 258
- [x] T0.3 평가 스크립트 (top-1 accuracy, 오답 덤프) → `src/evaluate.py` — 공식 가중 지표 0.7·한글+0.3·비한글
- [x] T0.4 베이스라인 holdout 점수 기록 (이후 모든 실험의 비교 기준) — **weighted 0.4890** (hangul 0.279 / non 0.979) → plans/P0.md

## P1. EDA
- [x] T1.1 컬럼별 결측률 (train, catalog) — GT/IGT 동시 결측 6.49%, flag 결측 16.49% → plans/eda/T1.1-T1.2.md
- [x] T1.2 정답 커버리지: target_vessel_id가 catalog에 항상 있는가, train 내 target 중복 빈도 — target 전부 catalog에 존재, 고유 5,851개(7.4%), target당 최대 4행 → 그룹 split 필요
- [x] T1.3 선박명 노이즈 유형 분류 (오타/띄어쓰기/약칭/한영혼용) — 불일치 100건 수기 샘플링 — 정규화 후 이름 일치 58.4%; 잔여는 한글음역 9.7%, 오타(누락/추가/전치) 31% → plans/eda/T1.3-T1.5.md
- [x] T1.4 톤수 비교: train gross_tonnage vs catalog 정답행 (gross_tonnage/DWT/size+unit 중 무엇과 맞는가, 오차 분포) — train GT ↔ catalog GT만 일치(정확 53.7%, 5% 이내 99%), ±5% 노이즈 → plans/eda/T1.4.md
- [x] T1.5 동명선 모호성: 정규화 이름이 같은 catalog 후보 수 분포 — catalog 동명선 0.79%, train 정답 중 동명선 0건
- [x] T1.6 flag_name · vessel_type 고유값 집합 (한글 ↔ 영문 매핑 대상 파악) — 국적 매핑 77% 일치(연도 의존), 선종 top-1 59.6% → 소프트 피처 → plans/eda/T1.6-T1.7.md
- [x] T1.7 call_sign이 catalog에 없다는 점 확인, 활용 가능성 판단 (train 내 동일 call_sign → 동일 target?) — call_sign↔target 1:1, 평가셋 겹침 미확인 → 보조용

## P2. 정규화
- [x] T2.1 선박명 normalizer (대소문자, 공백, 특수문자, 접두어 MV/MT 등) + **한글↔영문 음운 골격 키** → `src/normalize.py` — 한글 holdout top-1(단순 점수) 0.279→0.841
- [x] T2.2 flag 한글→영문 매핑표 — 수작업 표 대신 학습 P(영문|한글) soft 피처로 대체 (P4에서 fit)
- [x] T2.3 vessel_type 한글→영문 매핑표 — 동일, 학습 P(영문|한글) soft 피처 (P4)
- [x] T2.4 톤수 단위 통일 (catalog size/unit 해석) — 불필요(EDA T1.4: GT만 유효), `reference_tonnage` = 국제GT→GT
- [x] T2.5 normalizer 단위 테스트 (실제 오타 사례 10건 이상) — `python3 src/normalize.py` (실제 쌍 14건 assert)

## P3. 후보 생성 (목표: recall@K 최대화)
- [x] T3.1 정규화 이름 exact 인덱스 → `src/matcher.py`
- [x] T3.2 char n-gram TF-IDF 최근접 top-K — 3-gram 공유수 top-16 + **음운 골격 2-gram top-40** (TF-IDF 불필요)
- [x] T3.3 톤수 window 후보 (이름이 심하게 틀린 경우 보완) — GT/국제GT 최근접 각 12
- [x] T3.4 후보 합집합 + recall@K 측정 (K별 곡선, 놓친 케이스 분석) — holdout recall 한글 0.725→**0.985**, 비한글 0.983→0.991 (후보 52~64) → plans/P3.md

## P4. 재순위 모델
- [x] T4.1 쌍 피처 (이름 유사도 여러 종, 톤수 차이/비율, flag·type 일치, 결측 플래그) — 31개 + 후보집합 상대 7개 → `src/matcher.py` pair_features
- [x] T4.2 학습셋 빌더 (후보 negative, 그룹 단위 split) — Ranker.fit, 정답 미회수 시 강제 포함, ~61만 쌍
- [x] T4.3 LightGBM ranker 학습 및 holdout top-1 측정 — 평가 이미지에 LightGBM 없음 → sklearn HGB. `src/train.py` weighted 0.4890→**0.9592**
- [x] T4.4 오답 분석 → 피처 보강 루프 (1회 단위로 task 분리) — 1회차: 어두 X→S(병음), 모음 포함 로마자 유사도 피처 → weighted 0.9592→**0.9808** (hangul 0.946→0.977) → plans/P4.md

## P5. 견고성
- [x] T5.1 합성 노이즈(오타/결측) 증강으로 일반화 점검 — 증강은 clean 0.9808→0.9367로 역효과, 미채택. 오타 주입 시 0.9724 → plans/P5.md
- [x] T5.2 비공개 평가셋 분포 차이 가정 점검 (train에 없는 선박, 결측 증가) — holdout=전부 미등장 선박. 한글 톤수 결측 시 0.71이 위험이나 train 한글 결측 0건
- [x] T5.3 동명선 tie-break 규칙 (입항연도 등) — 불필요 (동명 원인 오답 0건)

## P6. 제출
- [x] T6.1 `submit/` 패키징 (script.py, model/, requirements.txt — baseline_submit 구조 준수) — `submit/`, `submit.zip`, sklearn 1.5.2로 학습한 pickle
- [x] T6.2 평가 서버 조건 로컬 재현 (클린 환경, 시간·메모리) — 3,000행 최대 34초, 383MB
- [x] T6.3 출력 검증 (sample_submission 형식, 3000행, 존재하는 vessel_id) — 통과 → plans/P6.md

## P7. 본선
- [x] T7.1 실험 로그 정리 (plans/ 기반 방법론 · 결과 · 실패 사례) → plans/P7.md (5-fold CV 0.9707 ± 0.017)
- [x] T7.2 현장 적용안 (후보 Top-N 추천 + 담당자 확인 워크플로) — 점수 ≥0.5 자동 확정 98.2%(정밀도 99.96%), 나머지 top-3 제시 → plans/P7.md

## P8. WORKLOG(v7 0.99722) 개선 아이디어 이식 실험 — plans/P8.md
- [~] T8.1 같은 언어모델 위에 LoRA 어댑터 앙상블 (정방향 + 역방향 + 시드)
- [~] T8.2 한글 행: 톤수 창 전체를 언어모델로 채점 (후보 누락 보완) — 계산량·재현율
- [x] T8.3 한글·영문 혼합 이름 키 — 합성 혼합명 골격 일치 0.580→0.927, 후보 recall 0.974→0.987, 기존 키 불변(84,441개 assert) → plans/scratch/p8_mixed/portable_patch.py
- [x] T8.4 카탈로그 쪽 사전확률 피처 (size/DWT 결측, 선종별 정답률, Unknown flag) — **기각**: 5-fold 0.9698→0.9669 (고침 3 / 틀어짐 7), 기존 결측 피처는 미사용 → plans/scratch/p8_feat/results.md
- [x] T8.5 입항 연도를 반영한 국적 피처 — **기각**: 0.9698→0.9699 (1행 바꿔치기, 노이즈)
- [x] T8.6 유형 단위 진단 제출 스크립트 (이분 탐색 금지) — 유형별 1등 답 진단은 식별 불가, **2등 후보 답 진단(D3)은 식별 가능**. 로컬 8개 구성 기대=실측 → plans/scratch/p8_diag/
- [x] T8.7 PYTHONHASHSEED 의존 버그 — 후보 동점을 인덱스로 고정, seed 0/1 holdout·제출 출력 동일, weighted 0.9808→0.9809, submit.zip 재생성 → plans/T8.7.md (vm_lib 점검은 별도)

## P9. 한글 행 개선 — plans/P9.md
- [x] T9.1 한글 행: ±10% 톤수 창에서 로마자·골격 유사도 상위 후보 추가 → `src/matcher.py` — 한글 recall 0.985→1.000, 5-fold 0.9707→**0.9841** (hangul 0.961→0.981)
- [x] T9.2 오답 분석 2회차 → 1자 삭제 색인 후보 + 선체 번호 피처 — 비한글 0.9923→**0.9984**, 5-fold 0.9841→**0.9851**
- [~] T9.3 언어모델(LoRA) 채점을 재순위 피처로 결합 — 코드·Colab 노트북 준비 완료 (`colab/p9_gpu_training.ipynb`), 5-fold 검증은 Colab에서 (기준 0.9851)
