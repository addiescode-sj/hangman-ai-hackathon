---
name: plan-first
description: Use before starting any TASK.md item. Enforces idea -> 3 options -> verification -> approval before touching src/ or submit/. Invoke as /plan-first <TASK ID>.
---

# Plan-First

Argument: a task ID from TASK.md (e.g. `T3.2`). If missing, list unchecked tasks and ask.

## Steps

1. **Set context.** Read TASK.md and CLAUDE.md. Write the ID to `.claude/current_task`. Mark the task `[~]`.
2. **Draft plan.** Copy `template.md` (this folder) to `plans/<ID>.md`. Fill Goal, Inputs, and 3 options:
   - A conservative / B balanced / C ambitious (names may differ).
   - Per option: approach, expected gain (quantified vs the T0.4 baseline, with a stated basis), practical benefit (runtime, maintainability, reuse by later tasks), cost, main risk.
   - Recommend one with a 2-line reason. Prefer evidence over intuition: run a cheap measurement first if it makes the gain estimate concrete.
3. **User picks.** Show the comparison table and recommendation; ask which option to take (AskUserQuestion). Record the choice.
4. **Verify.** Test the chosen option's riskiest assumption with the smallest experiment (scratch only, never in `src/`). Record command, numbers, and verdict in the plan's Verification section. If it fails, revise the plan or pick another option; do not proceed.
5. **Gate.** Only when verification passes, set `Status: VERIFIED` in the plan. The hook `plan-gate.sh` blocks edits to `src/` and `submit/` until then.
6. **Implement** exactly the verified scope. Measure on the fixed holdout.
7. **Close.** Tick the task in TASK.md with the artifact path and metric delta; set plan `Status: DONE`.

## Rules
- Never write implementation before step 5.
- Verification numbers must come from real runs; say so if a step was skipped.
- Scope creep (new idea found mid-task) becomes a new line in TASK.md, not part of this task.
