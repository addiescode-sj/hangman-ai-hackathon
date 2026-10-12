#!/usr/bin/env bash
# Block edits to src/ and submit/ unless the current task's plan is VERIFIED.
root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
path=$(jq -r '.tool_input.file_path // empty')
case "$path" in
  "$root"/src/*|"$root"/submit/*) ;;
  *) exit 0 ;;
esac
task=$(cat "$root/.claude/current_task" 2>/dev/null | tr -d "[:space:]")
if [ -z "$task" ]; then
  echo "BLOCKED: no active task. Run /plan-first <TASK ID> first." >&2; exit 2
fi
if ! grep -qE '^Status: (VERIFIED|DONE)' "$root/plans/$task.md" 2>/dev/null; then
  echo "BLOCKED: plans/$task.md is not VERIFIED. Finish options + verification (/plan-first $task) before implementing." >&2; exit 2
fi
