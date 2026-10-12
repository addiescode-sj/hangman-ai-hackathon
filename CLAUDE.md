# 선박 매칭 대회 — 작업 지침

입출항 기록 1행 → 카탈로그 `vessel_id` 1개 예측. 배경·규칙은 아래 요약, 로드맵은 [TASK.md](TASK.md).
데이터: `open/` (train.csv, vessel_catalog.csv, sample_submission.csv), 베이스라인: `baseline_submit/`.
제출은 코드 제출(데이콘 서버에서 비공개 평가셋으로 실행). 구조는 baseline_submit과 동일해야 한다.

## 필수 워크플로 (Plan-First)
모든 task는 **구현 전에** 계획 → 옵션 → 검증을 거친다. `/plan-first <TASK ID>` 로 시작.

1. 아이디어·구현 계획 작성 → `plans/<ID>.md`
2. 옵션 3개 비교 (예상 성능 향상 / 실질 반영 이점 / 비용·리스크) + 추천 1개
3. 사용자가 옵션 선택
4. 검증: 선택안의 핵심 가정을 작은 실험으로 확인 (결과 수치를 plan에 기록)
5. `Status: VERIFIED` 기록 후에만 `src/`, `submit/` 수정 가능 (hook이 강제)
6. 구현 → holdout 점수 측정 → TASK.md 체크 + 수치 기록

## 규칙
- 계획 단계 임시 실험은 scratchpad 또는 `plans/scratch/`에서 하고 `src/`에 넣지 않는다.
- 성능 비교는 항상 T0.2의 고정 holdout 기준. 기준 대비 증감을 기록한다.
- 한 번에 한 task. 현재 task ID는 `.claude/current_task` (plan-first가 설정).
- 코드와 주석은 영문으로 작성한다 (사용자 전역 지침). 문서(plan, TASK)는 한국어.
- 데이터 파일(`open/`)은 수정하지 않는다.
- 평가 서버는 비공개 데이터이므로 holdout 과적합 주의, 제출 전 T6.2·T6.3 필수.

## 디렉터리
- `plans/` 계획·검증 기록 (팀 공유, 커밋 대상)
- `src/` 개발 코드 · `submit/` 제출 패키지
- `.claude/skills/plan-first/` 워크플로 스킬 · `.claude/hooks/plan-gate.sh` 구현 게이트
