# TASK.md — 선박 매칭 로드맵

상태: `[ ]` 대기 · `[~]` 진행 · `[x]` 완료. 각 task는 **/plan-first <ID>** 로 시작한다 (계획 → 옵션 3개 → 검증 → 구현).
Task 완료 시 산출물 경로와 측정 수치를 한 줄로 덧붙인다.

## P0. 기준선 · 평가 체계
- [ ] T0.1 baseline_submit 실행 재현 (환경, 실행 시간, 출력 형식 확인)
- [ ] T0.2 holdout split 고정 (seed, 비율, train target 중복 처리 규칙) → `src/split.py`
- [ ] T0.3 평가 스크립트 (top-1 accuracy, 오답 덤프) → `src/evaluate.py`
- [ ] T0.4 베이스라인 holdout 점수 기록 (이후 모든 실험의 비교 기준)

## P1. EDA
- [ ] T1.1 컬럼별 결측률 (train, catalog)
- [ ] T1.2 정답 커버리지: target_vessel_id가 catalog에 항상 있는가, train 내 target 중복 빈도
- [ ] T1.3 선박명 노이즈 유형 분류 (오타/띄어쓰기/약칭/한영혼용) — 불일치 100건 수기 샘플링
- [ ] T1.4 톤수 비교: train gross_tonnage vs catalog 정답행 (gross_tonnage/DWT/size+unit 중 무엇과 맞는가, 오차 분포)
- [ ] T1.5 동명선 모호성: 정규화 이름이 같은 catalog 후보 수 분포
- [ ] T1.6 flag_name · vessel_type 고유값 집합 (한글 ↔ 영문 매핑 대상 파악)
- [ ] T1.7 call_sign이 catalog에 없다는 점 확인, 활용 가능성 판단 (train 내 동일 call_sign → 동일 target?)

## P2. 정규화
- [ ] T2.1 선박명 normalizer (대소문자, 공백, 특수문자, 접두어 MV/MT 등)
- [ ] T2.2 flag 한글→영문 매핑표
- [ ] T2.3 vessel_type 한글→영문 매핑표
- [ ] T2.4 톤수 단위 통일 (catalog size/unit 해석)
- [ ] T2.5 normalizer 단위 테스트 (실제 오타 사례 10건 이상)

## P3. 후보 생성 (목표: recall@K 최대화)
- [ ] T3.1 정규화 이름 exact 인덱스
- [ ] T3.2 char n-gram TF-IDF 최근접 top-K
- [ ] T3.3 톤수 window 후보 (이름이 심하게 틀린 경우 보완)
- [ ] T3.4 후보 합집합 + recall@K 측정 (K별 곡선, 놓친 케이스 분석)

## P4. 재순위 모델
- [ ] T4.1 쌍 피처 (이름 유사도 여러 종, 톤수 차이/비율, flag·type 일치, 결측 플래그)
- [ ] T4.2 학습셋 빌더 (후보 negative, 그룹 단위 split)
- [ ] T4.3 LightGBM ranker 학습 및 holdout top-1 측정
- [ ] T4.4 오답 분석 → 피처 보강 루프 (1회 단위로 task 분리)

## P5. 견고성
- [ ] T5.1 합성 노이즈(오타/결측) 증강으로 일반화 점검
- [ ] T5.2 비공개 평가셋 분포 차이 가정 점검 (train에 없는 선박, 결측 증가)
- [ ] T5.3 동명선 tie-break 규칙 (입항연도 등)

## P6. 제출
- [ ] T6.1 `submit/` 패키징 (script.py, model/, requirements.txt — baseline_submit 구조 준수)
- [ ] T6.2 평가 서버 조건 로컬 재현 (클린 환경, 시간·메모리)
- [ ] T6.3 출력 검증 (sample_submission 형식, 3000행, 존재하는 vessel_id)

## P7. 본선
- [ ] T7.1 실험 로그 정리 (plans/ 기반 방법론 · 결과 · 실패 사례)
- [ ] T7.2 현장 적용안 (후보 Top-N 추천 + 담당자 확인 워크플로)
