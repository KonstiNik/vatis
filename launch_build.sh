#!/usr/bin/env bash
# Launch an unattended Claude Code session that builds vatis end-to-end.
#
# Usage:
#   bash launch_build.sh              # start background build in tmux session 'vatis-build'
#   bash launch_build.sh attach       # attach to a running build (Ctrl-b d to detach)
#   bash launch_build.sh status       # show whether the session is alive
#   bash launch_build.sh log          # tail the build log
#   bash launch_build.sh stop         # kill the background session
#
# After launch:
#   - SESSION_SUMMARY.md   what got built (written by the agent at the end)
#   - BLOCKED.md           what got blocked, if anything (written when first blocked)
#   - build.log            full stdout/stderr stream

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="vatis-build"
LOG="${PROJECT_DIR}/build.log"

cmd="${1:-start}"

case "$cmd" in
  attach)
    tmux attach -t "$SESSION"
    exit 0
    ;;
  status)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "session '$SESSION' is RUNNING"
      tmux list-panes -t "$SESSION" -F '  pane #{pane_index}: #{pane_current_command}'
    else
      echo "session '$SESSION' is NOT running"
    fi
    if [ -f "$LOG" ]; then
      echo "log: $LOG ($(wc -l < "$LOG") lines, $(du -h "$LOG" | cut -f1))"
    fi
    if [ -f "${PROJECT_DIR}/BLOCKED.md" ]; then
      echo "BLOCKED.md exists — agent has logged at least one obstacle"
    fi
    if [ -f "${PROJECT_DIR}/SESSION_SUMMARY.md" ]; then
      echo "SESSION_SUMMARY.md exists — agent has finished (or last run finished)"
    fi
    exit 0
    ;;
  log)
    tail -f "$LOG"
    exit 0
    ;;
  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      tmux kill-session -t "$SESSION"
      echo "killed session '$SESSION'"
    else
      echo "no session '$SESSION' to kill"
    fi
    exit 0
    ;;
  start)
    ;;
  *)
    echo "unknown command: $cmd" >&2
    echo "usage: $0 [start|attach|status|log|stop]" >&2
    exit 1
    ;;
esac

# --- start ---

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' is already running. use 'attach' or 'stop'." >&2
  exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "error: 'claude' CLI not found in PATH" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "error: 'tmux' not found" >&2
  exit 1
fi

if [ ! -d "${PROJECT_DIR}/.venv" ]; then
  echo "error: ${PROJECT_DIR}/.venv does not exist" >&2
  echo "create the venv first (uv venv) so the agent can install into it" >&2
  exit 1
fi

if [ ! -f "${PROJECT_DIR}/CLAUDE.md" ]; then
  echo "error: CLAUDE.md not found at project root — cannot launch without spec" >&2
  exit 1
fi

# Archive any prior log so we don't tee into a stale file.
if [ -f "$LOG" ]; then
  mv "$LOG" "${LOG}.$(date +%Y%m%d-%H%M%S).bak"
fi

read -r -d '' PROMPT <<'PROMPT' || true
Build the vatis package end-to-end following CLAUDE.md.

Read CLAUDE.md first — it is the complete specification (math, architecture,
implementation order, conventions, permission model, unattended-mode rules).
Follow the implementation order in the "Implementation order" section.

Work autonomously. Use subagents (Explore, Plan, general-purpose) where they
help, per the "Working conventions" section. All installs go through uv into
the existing .venv at the project root — never global, never --user.

If you hit a permission denial: do not retry, try an allowed alternative, and
if none exists log it to BLOCKED.md and continue with independent work. Same
for any other obstacle you cannot resolve in a few attempts. Never use
destructive workarounds, never edit settings.local.json to self-grant.

Cross-validate every observable against perspic per the "tests/cross_validation/"
description in CLAUDE.md. Perspic is the reference implementation; any
disagreement beyond the documented tolerance is a bug in vatis.

When you finish (or hit a hard stop), write SESSION_SUMMARY.md at the project
root with: what was built, what works, what is tested, what is blocked, and
what the user should look at first when they return.
PROMPT

echo "launching background build in tmux session '$SESSION'..."
echo "log: $LOG"
echo

tmux new-session -d -s "$SESSION" -c "$PROJECT_DIR" \
  "claude -p \"\$PROMPT\" 2>&1 | tee \"$LOG\"; echo; echo '=== build finished ==='; exec bash"

# Pass the prompt into the tmux session's environment so the heredoc above is
# expanded inside the new shell rather than at script-eval time.
tmux set-environment -t "$SESSION" PROMPT "$PROMPT"
tmux respawn-pane -t "$SESSION" -k \
  "claude -p \"\$PROMPT\" 2>&1 | tee \"$LOG\"; echo; echo '=== build finished ==='; exec bash"

sleep 1
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' started."
  echo
  echo "useful commands:"
  echo "  bash launch_build.sh attach    # watch live (Ctrl-b d to detach)"
  echo "  bash launch_build.sh log       # tail the log"
  echo "  bash launch_build.sh status    # check whether it is still running"
  echo "  bash launch_build.sh stop      # kill it"
  echo
  echo "when the agent finishes it will write SESSION_SUMMARY.md."
  echo "if it gets blocked it will write BLOCKED.md (read this first when you return)."
else
  echo "failed to start tmux session" >&2
  exit 1
fi
