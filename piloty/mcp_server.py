"""MCP server interface for PiloTY.

Contract (agent-facing):
- `outcome` is the result of this tool call: `success`, `deadline_exceeded`,
  `eof`, `error`, `invalid_session`, or `terminated`.
- `terminal_state` is a best-effort interpretation of the rendered terminal after
  the tool finishes: `running`, `ready`, `password`, `confirm`, `repl`, `editor`,
  `pager`, or `unknown`.
- `output` is the PTY stream consumed during this call only (typically
  ANSI-stripped by default via `strip_ansi=true`).
- Use `snapshot_screen()` / `snapshot_scrollback()` for rendered views.

State interpretation uses heuristics by default. If the MCP client advertises
sampling capability, this server may use sampling to classify terminal state.
"""

import signal as signal_mod
import sys
import logging
import os
import asyncio
import re
import time
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase
from mcp.types import SamplingMessage, TextContent
from pydantic import ConfigDict, Field

from .core import PTY, default_session_log_dir

logger = logging.getLogger(__name__)

QUIESCENCE_MS = int(os.getenv("PILOTY_QUIESCENCE_MS", "1000"))
MAX_TOOL_TIMEOUT_S = 300.0

# FastMCP's generated argument models inherit from ArgModelBase. By default, extra
# tool arguments are silently ignored by pydantic. Reject unknown keys to avoid
# clients thinking unsupported parameters were applied.
ArgModelBase.model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

ANSI_RE = re.compile(
    r"(?:\x1b\[[0-?]*[ -/]*[@-~])|(?:\x1b\][^\x07]*(?:\x07|\x1b\\))",
    re.MULTILINE,
)

ESC_RE = re.compile(
    r"(?:\x1b[][()#][0-9A-Za-z])|(?:\x1b[=><])|(?:\x1b[78M])",
    re.MULTILINE,
)


def _clamp_tool_timeout(timeout: float) -> float:
    return min(timeout, MAX_TOOL_TIMEOUT_S)


def _maybe_strip_ansi(text: str, *, strip_ansi: bool) -> str:
    if not strip_ansi:
        return text
    s = ANSI_RE.sub("", text)
    s = ESC_RE.sub("", s)
    # Drop common control chars (BEL, etc) but keep newline, carriage return, tab.
    s = "".join(ch for ch in s if ch in {"\n", "\r", "\t", "\b"} or (ord(ch) >= 0x20 and ord(ch) != 0x7F))

    # Best-effort line normalization for carriage-return and backspace overstrike.
    out_lines: list[str] = []
    line: list[str] = []
    cursor = 0

    def flush_line(newline: bool):
        nonlocal line, cursor
        if newline or line:
            out_lines.append("".join(line).rstrip())
        line = []
        cursor = 0

    for ch in s:
        if ch == "\n":
            flush_line(newline=True)
            continue
        if ch == "\r":
            cursor = 0
            continue
        if ch == "\b":
            cursor = max(0, cursor - 1)
            continue
        if ch == "\t":
            ch = " "

        while len(line) <= cursor:
            line.append(" ")
        line[cursor] = ch
        cursor += 1

    if line:
        flush_line(newline=False)
    return "\n".join(out_lines).rstrip("\n")


def _terminal_state_from_state(state: str) -> str:
    if state == "READY":
        return "ready"
    if state == "PASSWORD":
        return "password"
    if state == "CONFIRM":
        return "confirm"
    if state == "REPL":
        return "repl"
    if state == "EDITOR":
        return "editor"
    if state == "PAGER":
        return "pager"
    if state == "RUNNING":
        return "running"
    return "unknown"


def _outcome_from_result_status(result_status: str | None) -> str:
    if result_status in {"quiescent", "matched"}:
        return "success"
    if result_status == "timeout":
        return "deadline_exceeded"
    if result_status == "eof":
        return "eof"
    if result_status == "error":
        return "error"
    return "error"


async def _describe_terminal(session: PTY, ctx: Context | None) -> tuple[str | None, str]:
    if not session.alive:
        return None, "pty eof"
    snap = await asyncio.to_thread(session.screen_snapshot, drain=False)
    state, reason = await determine_terminal_state(
        ctx,
        snap["screen"],
        cursor_x=snap.get("cursor_x"),
        shell_prompt_regex=getattr(session, "shell_prompt_regex", None),
    )
    return _terminal_state_from_state(state), reason


async def _build_stream_response(
    *,
    session: PTY,
    ctx: Context | None,
    result: dict,
    strip_ansi: bool,
    output: str | None = None,
    extra: dict | None = None,
) -> dict:
    terminal_state, state_reason = await _describe_terminal(session, ctx)
    resp = {
        "outcome": _outcome_from_result_status(result.get("status")),
        "terminal_state": terminal_state,
        "state_reason": state_reason,
        "output": _maybe_strip_ansi(result.get("output", "") if output is None else output, strip_ansi=strip_ansi),
        "output_truncated": bool(result.get("output_truncated", False)),
        "dropped_bytes": int(result.get("dropped_bytes", 0)),
    }
    if extra:
        resp.update(extra)
    if result.get("status") in {"error", "eof"}:
        resp["error"] = str(result.get("error", "pty eof"))
    return resp


def _configure_logging():
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    log_path = "/tmp/piloty.log"
    try:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return
    except Exception:
        pass

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


TERMINAL_STATE_PROMPT = """Analyze this terminal screen and determine its state.

Screen content:
```
{screen}
```

Classification rules:
- If a shell prompt / REPL prompt / editor / pager is visible (especially on the last line),
  choose that state even if earlier lines contain errors (e.g., tracebacks).
- Use ERROR only when the screen looks stuck in an error state with no interactive prompt visible.

What is the terminal state? Answer with exactly one of:
- READY: Shell prompt visible, waiting for command (e.g., $, #, >, PS1)
- PASSWORD: Asking for password (e.g., "Password:", "Enter passphrase")
- CONFIRM: Asking for confirmation (e.g., "[Y/n]", "Continue?", "Are you sure?")
- REPL: Interactive interpreter prompt (e.g., ">>>", "In [1]:", "(Pdb)", "ipdb>", "irb>", "mysql>")
- EDITOR: Text editor active (e.g., vim, nano, emacs)
- PAGER: Pager active (e.g., less, more, man page)
- RUNNING: Command still executing, no prompt visible
- ERROR: Error message visible
- UNKNOWN: Cannot determine state

Respond with just the state name and a brief reason, e.g.:
READY: bash prompt visible
PASSWORD: SSH asking for password
CONFIRM: apt asking to continue"""


class SessionManager:
    """Manages multiple PTY instances."""

    def __init__(self):
        self.sessions: dict[str, PTY] = {}
        self._last_used: dict[str, float] = {}
        self._max_sessions: int = 32
        self._terminated: set[str] = set()
        self._config: dict[str, dict] = {}

    def get_session(self, session_id: str, *, cwd: str | None = None) -> PTY:
        """Get or create PTY session."""
        if session_id in self._terminated:
            raise RuntimeError("terminated")

        existing = self.sessions.get(session_id)
        if existing is not None:
            self._last_used[session_id] = time.monotonic()
            return existing

        if session_id not in self.sessions:
            if self._max_sessions > 0 and len(self.sessions) >= self._max_sessions:
                oldest_id = min(self._last_used, key=self._last_used.get, default=None)
                if oldest_id is not None:
                    try:
                        self.sessions[oldest_id].terminate()
                    except Exception:
                        pass
                    self.sessions.pop(oldest_id, None)
                    self._last_used.pop(oldest_id, None)
            if cwd is None:
                raise ValueError("cwd is required when creating a new session")
            cfg = self._config.get(session_id, {})
            self.sessions[session_id] = PTY(
                session_id=session_id,
                cwd=cwd,
                shell_prompt_regex=cfg.get("shell_prompt_regex"),
                description=cfg.get("description"),
            )
        self._last_used[session_id] = time.monotonic()
        return self.sessions[session_id]

    def configure(
        self,
        session_id: str,
        *,
        description: str | None = None,
        shell_prompt_regex: str | None = None,
    ):
        cfg = self._config.setdefault(session_id, {})
        if description is not None:
            cfg["description"] = description
        if shell_prompt_regex is not None:
            cfg["shell_prompt_regex"] = shell_prompt_regex

        s = self.sessions.get(session_id)
        if s is not None:
            if description is not None:
                s.description = description
            if shell_prompt_regex is not None:
                s.shell_prompt_regex = shell_prompt_regex

    def configure_full(
        self,
        session_id: str,
        *,
        description: str | None = None,
        shell_prompt_regex: str | None = None,
    ):
        if session_id in self._terminated:
            raise RuntimeError("terminated")

        cfg = self._config.setdefault(session_id, {})
        if description is not None:
            cfg["description"] = description
        if shell_prompt_regex is not None:
            cfg["shell_prompt_regex"] = shell_prompt_regex

        s = self.sessions.get(session_id)
        if s is not None:
            if description is not None:
                s.description = description
            if shell_prompt_regex is not None:
                s.shell_prompt_regex = shell_prompt_regex

    def list_sessions(self) -> list[dict]:
        ids = set(self.sessions.keys()) | set(self._terminated) | set(self._config.keys())
        out: list[dict] = []
        for session_id in sorted(ids):
            terminated = session_id in self._terminated
            s = self.sessions.get(session_id)
            meta = None
            alive = False
            if s is not None:
                alive = bool(s.alive)
                try:
                    meta = s.metadata()
                except Exception:
                    meta = None
            out.append(
                {
                    "session_id": session_id,
                    "terminated": terminated,
                    "alive": alive,
                    "description": (meta or {}).get("description"),
                    "cwd": (meta or {}).get("cwd"),
                    "pid": (meta or {}).get("pid"),
                    "shell_prompt_regex": (meta or {}).get("shell_prompt_regex"),
                    "rows": (meta or {}).get("rows"),
                    "cols": (meta or {}).get("cols"),
                    "started_at": (meta or {}).get("started_at"),
                    "last_activity_at": (meta or {}).get("last_activity_at"),
                }
            )
        return out

    def terminate_all(self):
        """Terminate all sessions."""
        for session_id, session in list(self.sessions.items()):
            session.terminate()
            self.sessions.pop(session_id, None)
            self._last_used.pop(session_id, None)
            self._terminated.add(session_id)


# Initialize MCP server
mcp = FastMCP("PiloTY", dependencies=["pexpect", "pyte"])
session_manager = SessionManager()


async def interpret_terminal_state(
    ctx: Context,
    screen: str,
) -> tuple[str, str]:
    """Use LLM sampling to interpret terminal state.

    Returns:
        (state, reason) tuple
    """
    try:
        result = await ctx.session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=TERMINAL_STATE_PROMPT.format(screen=screen),
                    ),
                )
            ],
            max_tokens=50,
        )

        if result.content.type == "text":
            response = result.content.text.strip()
            if ":" in response:
                state, reason = response.split(":", 1)
                return state.strip().upper(), reason.strip()
            m = re.search(r"\b(READY|PASSWORD|CONFIRM|REPL|EDITOR|PAGER|RUNNING|ERROR|UNKNOWN)\b", response.upper())
            if m:
                state = m.group(1)
                reason = response[m.end() :].strip(" \t\r\n:-")
                return state, reason
            return "UNKNOWN", response[:100]
        return "UNKNOWN", "sampling failed"
    except Exception as e:
        logger.warning(f"Sampling failed: {e}")
        return "UNKNOWN", str(e)


async def determine_terminal_state(
    ctx: Context | None,
    screen: str,
    cursor_x: int | None = None,
    shell_prompt_regex: str | None = None,
) -> tuple[str, str]:
    """Determine terminal state using sampling when available, else heuristics.

    If sampling is requested but fails or yields UNKNOWN, fall back to heuristics
    and include the sampling reason in the returned reason for troubleshooting.
    """
    heuristic_state, heuristic_reason = detect_state_heuristic(
        screen,
        cursor_x=cursor_x,
        shell_prompt_regex=shell_prompt_regex,
    )
    if ctx and getattr(ctx, "session", None):
        # Client sampling is optional in MCP. Some clients provide a session object
        # but do not advertise sampling capability, in which case create_message()
        # will fail with "sampling/createMessage". Treat that as "no sampling"
        # and rely on local heuristics instead.
        client_params = getattr(ctx.session, "client_params", None)
        caps = getattr(client_params, "capabilities", None) if client_params else None
        sampling_caps = getattr(caps, "sampling", None) if caps else None
        if sampling_caps is None:
            return heuristic_state, heuristic_reason

        # Heuristic-first. Sampling is used only to refine RUNNING into a more
        # specific interactive state. This prevents sampling from incorrectly
        # reporting READY when the screen shows a running command line.
        if heuristic_state != "RUNNING":
            return heuristic_state, heuristic_reason

        sampled_state, sampled_reason = await interpret_terminal_state(ctx, screen)
        if sampled_state in {"PASSWORD", "CONFIRM", "REPL", "EDITOR", "PAGER"}:
            return sampled_state, sampled_reason

        if sampled_state == "UNKNOWN":
            return (
                heuristic_state,
                f"sampling=UNKNOWN ({sampled_reason}); heuristic={heuristic_state} ({heuristic_reason})",
            )
        return heuristic_state, f"sampling={sampled_state} ({sampled_reason}); heuristic={heuristic_reason}"

    return heuristic_state, heuristic_reason


def detect_state_heuristic(
    screen: str,
    *,
    cursor_x: int | None = None,
    shell_prompt_regex: str | None = None,
) -> tuple[str, str]:
    """Fast heuristic state detection (no LLM).

    Used as fallback when sampling unavailable.
    """
    lines = screen.strip().split("\n")
    if not lines:
        return "UNKNOWN", "empty screen"

    # Heuristics should not get "stuck" on scrollback text. Prefer signals near the
    # bottom of the visible screen.
    window = lines[-12:]
    window_lower = "\n".join(window).lower()
    tail_nonempty = [ln.rstrip() for ln in window if ln.strip()]
    tail_last = tail_nonempty[-1] if tail_nonempty else lines[-1].rstrip()

    # REPL prompts - check exact patterns
    # Use tail window to preserve case and spacing.
    repl_patterns = [
        (">>> ", "python"),
        (">>>", "python"),  # Also match without trailing space
        ("... ", "python continuation"),
        ("in [", "ipython"),
        ("out[", "ipython output"),
        ("(pdb)", "pdb"),
        ("ipdb>", "ipdb"),
        ("irb(", "ruby"),
        ("pry(", "pry"),
        ("mysql>", "mysql"),
        ("postgres=#", "psql"),
        ("postgres=>", "psql"),
        ("sqlite>", "sqlite"),
    ]
    for prompt, name in repl_patterns:
        if (cursor_x is None or cursor_x > 0) and (prompt in tail_last or prompt in tail_last.lower()):
            return "REPL", f"{name} prompt"

    # Editor detection (vim, nano)
    if "-- insert --" in window_lower or "-- normal --" in window_lower:
        return "EDITOR", "vim mode indicator"
    if "gnu nano" in window_lower or "^g get help" in window_lower:
        return "EDITOR", "nano indicators"

    # Pager detection
    tail_last_lower = tail_last.lower()
    if tail_last == ":" or "(end)" in tail_last_lower or "manual page" in window_lower:
        return "PAGER", "pager indicators"

    # Configurable prompt detection (shell).
    if shell_prompt_regex:
        try:
            m = re.search(shell_prompt_regex, tail_last)
        except re.error:
            m = None
        if m:
            if cursor_x is not None and cursor_x == 0:
                return "RUNNING", "cursor at column 0"
            return "READY", f"shell_prompt_regex matched: {m.group(0)!r}"

    # Shell prompts - must look like actual prompts, not progress bars
    # Require typical prompt structure: ends with $ # or > but not inside brackets
    shell_ends = ["$", "#"]
    tail_last_stripped = tail_last.rstrip()
    for end in shell_ends:
        if tail_last_stripped.endswith(end):
            if "%" in tail_last_stripped or ("[" in tail_last_stripped and "]" in tail_last_stripped):
                break
            if cursor_x is not None and cursor_x == 0:
                return "RUNNING", "cursor at column 0"
            return "READY", f"shell prompt '{end}'"

    # Special case: bare > prompt (but not inside progress bars or with percentages)
    if tail_last_stripped.endswith(">"):
        if "%" not in tail_last_stripped and "[" not in tail_last_stripped:
            if len(tail_last_stripped) < 50:
                if cursor_x is not None and cursor_x == 0:
                    return "RUNNING", "cursor at column 0"
                return "READY", "generic prompt"

    # zsh prompt: ends with %
    if tail_last_stripped.endswith("%"):
        if not tail_last_stripped[-2:-1].isdigit():
            if cursor_x is not None and cursor_x == 0:
                return "RUNNING", "cursor at column 0"
            return "READY", "zsh prompt"

    # Password / confirmation prompts (only if no interactive prompt detected).
    # Password prompts: only consider the last few visible lines to avoid stale
    # "Password:" text in scrollback overriding the current state.
    pw_recent = tail_nonempty[-3:] if tail_nonempty else [tail_last]
    pw_recent_lower = "\n".join(ln.lower() for ln in pw_recent)
    password_patterns = [
        "password:",
        "password for",
        "passphrase:",
        "passphrase for",
        "enter password",
        "enter passphrase",
        "[sudo]",
        "secret:",
    ]
    for pattern in password_patterns:
        if pattern in pw_recent_lower:
            return "PASSWORD", "password prompt detected"

    # Confirmation prompts: also only consider the last few visible lines.
    confirm_indicators = ["[y/n]", "[yes/no]", "continue?", "are you sure", "proceed?"]
    for indicator in confirm_indicators:
        if indicator in pw_recent_lower:
            return "CONFIRM", f"found '{indicator}'"

    # Error detection (very low priority): only consider the last few visible lines.
    # This prevents stale scrollback exceptions from overriding the current state.
    if cursor_x is not None and cursor_x == 0:
        return "RUNNING", "cursor at column 0"

    recent = [ln.lower() for ln in tail_nonempty[-3:]] if tail_nonempty else [tail_last.lower()]
    recent_join = "\n".join(recent)
    error_indicators = ["error:", "failed:", "fatal:", "exception:", "traceback", "indexerror", "keyerror"]
    for indicator in error_indicators:
        if indicator in recent_join:
            return "ERROR", f"found '{indicator}'"

    return "RUNNING", "no prompt detected"


def _session_log_dir_exists(session_id: str) -> bool:
    return os.path.isdir(str(default_session_log_dir(session_id)))


def _session_transcript_path_if_exists(session_id: str) -> str | None:
    tp = str(default_session_log_dir(session_id) / "transcript.log")
    return tp if os.path.isfile(tp) else None


def _missing_session_hint(session_id: str) -> str:
    hint = "no such session_id; call create_session(session_id, cwd) first"
    if _session_log_dir_exists(session_id):
        hint = "no such session_id (server restarted or session evicted; logs exist on disk); call create_session(session_id, cwd) first"
    return hint


@mcp.tool()
async def create_session(
    session_id: str,
    cwd: str,
    description: Annotated[
        str | None,
        Field(
            description="Free-form description of what this session is doing (for humans). Does not affect execution."
        ),
    ] = None,
    shell_prompt_regex: Annotated[
        str | None,
        Field(
            description=(
                "Optional regex used to detect when the terminal is idle at a shell prompt. "
                "Matched against the last visible non-empty screen line. Prefer anchoring to the end."
            )
        ),
    ] = None,
    ctx: Context | None = None,
) -> dict:
    """Create a PTY session with an explicit working directory.

    MCP does not provide a standard field for the client's current working
    directory. The caller must provide `cwd` explicitly.

    Args:
        session_id: Stable identifier for this PTY session.
        cwd: Working directory for the session's shell.
        description: Free-form description of what this session is doing (for humans).
        shell_prompt_regex: Optional regex to detect a shell prompt on the last visible
            non-empty screen line. Use this only for shell prompt detection, not the LLM prompt.
            Prefer anchoring to the end of the prompt (e.g. `r\"\\$\\s*$\"`). Set it after
            observing the actual prompt text if the default heuristics misclassify READY as RUNNING.
    """
    if session_id in getattr(session_manager, "_terminated", set()):
        return {"outcome": "terminated", "terminal_state": None, "created": False, "state_reason": ""}

    if not cwd:
        raise ValueError("cwd must be a non-empty path")
    abs_cwd = os.path.abspath(cwd)
    if not os.path.isdir(abs_cwd):
        raise ValueError(f"cwd is not an existing directory: {abs_cwd}")

    session = session_manager.sessions.get(session_id)
    if session is None:
        try:
            session_manager.configure_full(
                session_id,
                description=description,
                shell_prompt_regex=shell_prompt_regex,
            )
            session = session_manager.get_session(session_id, cwd=abs_cwd)
        except RuntimeError:
            return {"outcome": "terminated", "terminal_state": None, "created": False, "state_reason": ""}
        created = True
        created_reason = "session created"
    else:
        meta = await asyncio.to_thread(session.metadata)
        existing_cwd = os.path.abspath(str(meta.get("cwd", "")))
        if existing_cwd and existing_cwd != abs_cwd:
            raise ValueError(
                f"session_id '{session_id}' already exists with cwd '{existing_cwd}', requested '{abs_cwd}'"
            )
        session_manager.configure_full(
            session_id,
            description=description,
            shell_prompt_regex=shell_prompt_regex,
        )
        created = False
        created_reason = "session already exists"

    terminal_state, reason = await _describe_terminal(session, ctx)
    meta = await asyncio.to_thread(session.metadata)
    return {
        "outcome": "success",
        "terminal_state": terminal_state,
        "created": created,
        "cwd": meta.get("cwd"),
        "description": meta.get("description"),
        "shell_prompt_regex": meta.get("shell_prompt_regex"),
        "state_reason": f"{created_reason}: {reason}",
    }


@mcp.tool()
async def send_line(
    session_id: str,
    line: str,
    deadline_s: float = 30.0,
    quiet_ms: int = QUIESCENCE_MS,
    strip_ansi: bool = True,
    ctx: Context | None = None,
) -> dict:
    """Send one newline-terminated line to a stateful PTY session.

    Requires an existing session created via `create_session(session_id, cwd)`.

    After sending `line + "\\n"`, this drains newly produced PTY bytes until the
    PTY is silent for `quiet_ms` milliseconds or until `deadline_s` expires.

    `deadline_s` is capped at 300 seconds.

    Common SSH pattern:
    - create_session(session_id, cwd)
    - send_line(session_id, "ssh host", deadline_s=2)
    - wait_for_shell_prompt(session_id, deadline_s=30)
    """
    if session_id in getattr(session_manager, "_terminated", set()):
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": "",
        }
    if session_id not in session_manager.sessions:
        return {
            "outcome": "invalid_session",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": _missing_session_hint(session_id),
        }
    try:
        session = session_manager.get_session(session_id)
    except RuntimeError:
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": "",
        }

    deadline_s = _clamp_tool_timeout(deadline_s)
    result = await asyncio.to_thread(session.type, line + "\n", timeout=deadline_s, quiescence_ms=quiet_ms)
    return await _build_stream_response(session=session, ctx=ctx, result=result, strip_ansi=strip_ansi)


@mcp.tool()
async def send_text(
    session_id: str,
    text: str,
    deadline_s: float = 30.0,
    quiet_ms: int = QUIESCENCE_MS,
    strip_ansi: bool = True,
    ctx: Context | None = None,
) -> dict:
    """Send raw text to a stateful PTY session without adding a newline.

    Requires an existing session created via `create_session(session_id, cwd)`.

    After sending `text`, this drains newly produced PTY bytes until the PTY is
    silent for `quiet_ms` milliseconds or until `deadline_s` expires.

    `deadline_s` is capped at 300 seconds.
    """
    if session_id in getattr(session_manager, "_terminated", set()):
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": "",
        }
    if session_id not in session_manager.sessions:
        return {
            "outcome": "invalid_session",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": _missing_session_hint(session_id),
        }
    try:
        session = session_manager.get_session(session_id)
    except RuntimeError:
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": "",
        }

    deadline_s = _clamp_tool_timeout(deadline_s)
    result = await asyncio.to_thread(session.type, text, timeout=deadline_s, quiescence_ms=quiet_ms)
    return await _build_stream_response(session=session, ctx=ctx, result=result, strip_ansi=strip_ansi)


@mcp.tool()
async def send_password(
    session_id: str,
    password: str,
    deadline_s: float = 30.0,
    quiet_ms: int = QUIESCENCE_MS,
    ctx: Context | None = None,
) -> dict:
    """Send a password plus newline.

    Requires an existing session created via `create_session(session_id, cwd)`.

    Security model:
    - Forces terminal echo off for this send operation (`echo=False`).
    - Disables transcript logging for this send operation (`log=False`).
    - Returns the PTY bytes consumed during this call with best-effort password
      redaction, prefixed by `[password sent]`.

    After sending `password + "\\n"`, this drains newly produced PTY bytes until
    the PTY is silent for `quiet_ms` milliseconds or until `deadline_s` expires.

    `deadline_s` is capped at 300 seconds.
    """
    if session_id in getattr(session_manager, "_terminated", set()):
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": "",
        }
    if session_id not in session_manager.sessions:
        return {
            "outcome": "invalid_session",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": _missing_session_hint(session_id),
        }
    try:
        session = session_manager.get_session(session_id)
    except RuntimeError:
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": "",
        }

    deadline_s = _clamp_tool_timeout(deadline_s)

    result = await asyncio.to_thread(
        session.type,
        password + "\n",
        timeout=deadline_s,
        quiescence_ms=quiet_ms,
        log=False,
        echo=False,
    )
    post = _maybe_strip_ansi(result.get("output", ""), strip_ansi=True)
    # Best-effort redact: some programs may still echo input even when echo is
    # disabled, or may include the password in error messages.
    redacted = post.replace(password, "[redacted]") if password else post
    if redacted.strip():
        out = "[password sent]\n" + redacted
    else:
        out = "[password sent]"
    return await _build_stream_response(session=session, ctx=ctx, result=result, strip_ansi=False, output=out)


@mcp.tool()
async def send_control(
    session_id: str,
    key: str,
    deadline_s: float = 5.0,
    quiet_ms: int = QUIESCENCE_MS,
    strip_ansi: bool = True,
    ctx: Context | None = None,
) -> dict:
    """Send a control character to the terminal.

    Requires an existing session created via `create_session(session_id, cwd)`.

    Keys:
    - `c` sends Ctrl+C, commonly used to interrupt.
    - `d` sends Ctrl+D (EOF in many programs).
    - `z` sends Ctrl+Z (job control suspend).
    - `l` sends Ctrl+L (clear screen in many shells).
    - `[` or `escape` sends ESC.

    After sending the control character, this drains newly produced PTY bytes
    until the PTY is silent for `quiet_ms` milliseconds or until `deadline_s`
    expires.

    `deadline_s` is capped at 300 seconds.
    """
    if session_id in getattr(session_manager, "_terminated", set()):
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": "",
        }
    if session_id not in session_manager.sessions:
        return {
            "outcome": "invalid_session",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": _missing_session_hint(session_id),
        }
    try:
        session = session_manager.get_session(session_id)
    except RuntimeError:
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": "",
        }

    # Map key to control character
    key = key.lower()
    if key == "[" or key == "escape" or key == "esc":
        char = "\x1b"  # Escape
    elif len(key) == 1 and key.isalpha():
        char = chr(ord(key) - ord("a") + 1)  # Ctrl+letter
    else:
        raise ValueError(f"Unknown control key: {key}")

    deadline_s = _clamp_tool_timeout(deadline_s)
    result = await asyncio.to_thread(session.type, char, timeout=deadline_s, quiescence_ms=quiet_ms)
    return await _build_stream_response(session=session, ctx=ctx, result=result, strip_ansi=strip_ansi)


@mcp.tool()
async def wait_for_output(
    session_id: str,
    deadline_s: float = 30.0,
    quiet_ms: int = QUIESCENCE_MS,
    strip_ansi: bool = True,
    ctx: Context | None = None,
) -> dict:
    """Wait for newly arriving PTY output without sending input.

    Requires an existing session created via `create_session(session_id, cwd)`.

    This waits for new PTY bytes to arrive. After the first new output is
    observed, it continues draining until the PTY is silent for `quiet_ms`
    milliseconds or until `deadline_s` expires.

    `deadline_s` is capped at 300 seconds. If the deadline expires after some
    output has already been captured, partial output is returned with
    `outcome="deadline_exceeded"`.

    If you need to wait specifically for a shell prompt, use
    `wait_for_shell_prompt()`.
    """
    if session_id in getattr(session_manager, "_terminated", set()):
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": "",
        }
    if session_id not in session_manager.sessions:
        return {
            "outcome": "invalid_session",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": _missing_session_hint(session_id),
        }
    try:
        session = session_manager.get_session(session_id)
    except RuntimeError:
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": "",
        }

    deadline_s = _clamp_tool_timeout(deadline_s)
    result = await asyncio.to_thread(session.poll_output, timeout=deadline_s, quiescence_ms=quiet_ms)
    return await _build_stream_response(session=session, ctx=ctx, result=result, strip_ansi=strip_ansi)


@mcp.tool()
async def snapshot_screen(session_id: str, ctx: Context | None = None) -> dict:
    """Get the current VT100-rendered terminal screen snapshot.

    Requires an existing session created via `create_session(session_id, cwd)`.

    This is a passive snapshot. It does not ingest new PTY bytes.
    """
    if session_id in getattr(session_manager, "_terminated", set()):
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "screen": "",
            "cursor_x": None,
            "cursor_y": None,
            "vt100_ok": None,
            "rows": None,
            "cols": None,
            "state_reason": "",
        }
    if session_id not in session_manager.sessions:
        return {
            "outcome": "invalid_session",
            "terminal_state": None,
            "screen": "",
            "cursor_x": None,
            "cursor_y": None,
            "vt100_ok": None,
            "rows": None,
            "cols": None,
            "state_reason": _missing_session_hint(session_id),
        }
    session = session_manager.sessions[session_id]

    snap = await asyncio.to_thread(session.screen_snapshot, drain=False)
    terminal_state, reason = await _describe_terminal(session, ctx)
    return {
        "outcome": "success",
        "terminal_state": terminal_state,
        "screen": snap["screen"],
        "cursor_x": snap.get("cursor_x"),
        "cursor_y": snap.get("cursor_y"),
        "vt100_ok": snap.get("vt100_ok"),
        "rows": snap.get("rows"),
        "cols": snap.get("cols"),
        "state_reason": reason,
    }


@mcp.tool()
async def snapshot_scrollback(
    session_id: str,
    lines: int = 200,
    strip_ansi: bool = False,
    ctx: Context | None = None,
) -> dict:
    """Get rendered terminal scrollback.

    Requires an existing session created via `create_session(session_id, cwd)`.

    This is a passive snapshot. It does not ingest new PTY bytes.
    """
    if session_id in getattr(session_manager, "_terminated", set()):
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "scrollback": "",
            "rows": None,
            "cols": None,
            "state_reason": "",
        }
    if session_id not in session_manager.sessions:
        return {
            "outcome": "invalid_session",
            "terminal_state": None,
            "scrollback": "",
            "rows": None,
            "cols": None,
            "state_reason": _missing_session_hint(session_id),
        }
    session = session_manager.sessions[session_id]

    snap = await asyncio.to_thread(session.screen_snapshot, drain=False)
    terminal_state, reason = await _describe_terminal(session, ctx)
    scrollback = await asyncio.to_thread(session.get_scrollback, lines, drain=False)
    scrollback = _maybe_strip_ansi(scrollback, strip_ansi=strip_ansi)
    return {
        "outcome": "success",
        "terminal_state": terminal_state,
        "scrollback": scrollback,
        "rows": snap.get("rows"),
        "cols": snap.get("cols"),
        "state_reason": reason,
    }


@mcp.tool()
async def clear_scrollback(session_id: str, ctx: Context | None = None) -> dict:
    """Clear rendered scrollback history while preserving current screen.

    Requires an existing session created via `create_session(session_id, cwd)`.
    """
    if session_id in getattr(session_manager, "_terminated", set()):
        return {"outcome": "terminated", "terminal_state": None, "state_reason": ""}
    if session_id not in session_manager.sessions:
        return {"outcome": "invalid_session", "terminal_state": None, "state_reason": _missing_session_hint(session_id)}
    session = session_manager.sessions[session_id]

    await asyncio.to_thread(session.clear_scrollback)
    terminal_state, reason = await _describe_terminal(session, ctx)
    return {"outcome": "success", "terminal_state": terminal_state, "state_reason": reason}


@mcp.tool()
async def wait_for_regex(
    session_id: str,
    pattern: str,
    deadline_s: float = 30.0,
    search_scope: str = "visible_or_new",
    strip_ansi: bool = True,
    ctx: Context | None = None,
) -> dict:
    """Wait for a regex match in rendered text and/or newly arriving PTY bytes.

    Requires an existing session created via `create_session(session_id, cwd)`.

    `search_scope` controls where matching begins:
    - `visible_or_new`: check already rendered scrollback first, then wait on new PTY output.
    - `new_only`: ignore already visible text and wait only on new PTY output.

    `deadline_s` is capped at 300 seconds.
    """
    if search_scope not in {"visible_or_new", "new_only"}:
        raise ValueError("search_scope must be one of: visible_or_new, new_only")
    if session_id in getattr(session_manager, "_terminated", set()):
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "output": "",
            "matched": False,
            "match": None,
            "groups": [],
            "match_source": None,
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": "",
        }
    if session_id not in session_manager.sessions:
        return {
            "outcome": "invalid_session",
            "terminal_state": None,
            "output": "",
            "matched": False,
            "match": None,
            "groups": [],
            "match_source": None,
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": _missing_session_hint(session_id),
        }
    try:
        session = session_manager.get_session(session_id)
    except RuntimeError:
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "output": "",
            "matched": False,
            "match": None,
            "groups": [],
            "match_source": None,
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": "",
        }

    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"re.error: {e}")

    if search_scope == "visible_or_new":
        visible = await asyncio.to_thread(session.get_scrollback, 5000, log=False, drain=False)
        visible_match = rx.search(visible)
        if visible_match:
            terminal_state, reason = await _describe_terminal(session, ctx)
            return {
                "outcome": "success",
                "terminal_state": terminal_state,
                "output": "",
                "matched": True,
                "match": visible_match.group(0),
                "groups": list(visible_match.groups()),
                "match_source": "visible",
                "output_truncated": False,
                "dropped_bytes": 0,
                "state_reason": f"matched on rendered text: {reason}",
            }

    deadline_s = _clamp_tool_timeout(deadline_s)
    result = await asyncio.to_thread(session.expect, pattern, deadline_s)
    resp = await _build_stream_response(
        session=session,
        ctx=ctx,
        result=result,
        strip_ansi=strip_ansi,
        extra={
            "matched": result.get("status") == "matched",
            "match": result.get("match"),
            "groups": result.get("groups", []),
            "match_source": "new_output" if result.get("status") == "matched" else None,
        },
    )
    return resp


@mcp.tool()
async def wait_for_shell_prompt(
    session_id: str,
    deadline_s: float = 30.0,
    quiet_ms: int = QUIESCENCE_MS,
    strip_ansi: bool = True,
    ctx: Context | None = None,
) -> dict:
    """Wait until a shell prompt is detected.

    Requires an existing session created via `create_session(session_id, cwd)`.

    This repeatedly ingests new PTY output and re-runs prompt detection until a
    shell prompt is visible or until `deadline_s` expires. Any output consumed
    during that wait is returned in `output`.

    If the remote shell uses a customized prompt that the default heuristics
    misclassify, set `shell_prompt_regex` via `configure_session(...)` or
    `create_session(...)`.

    `deadline_s` is capped at 300 seconds.
    """
    if session_id in getattr(session_manager, "_terminated", set()):
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "output": "",
            "prompt_detected": False,
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": "",
        }
    if session_id not in session_manager.sessions:
        return {
            "outcome": "invalid_session",
            "terminal_state": None,
            "output": "",
            "prompt_detected": False,
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": _missing_session_hint(session_id),
        }

    session = session_manager.sessions[session_id]
    terminal_state, reason = await _describe_terminal(session, ctx)
    if terminal_state == "ready":
        return {
            "outcome": "success",
            "terminal_state": terminal_state,
            "output": "",
            "prompt_detected": True,
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": reason,
        }

    deadline_s = _clamp_tool_timeout(deadline_s)
    deadline = time.monotonic() + deadline_s
    chunks: list[str] = []
    dropped_bytes = 0
    output_truncated = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            terminal_state, reason = await _describe_terminal(session, ctx)
            return {
                "outcome": "deadline_exceeded",
                "terminal_state": terminal_state,
                "output": _maybe_strip_ansi("".join(chunks), strip_ansi=strip_ansi),
                "prompt_detected": False,
                "output_truncated": output_truncated,
                "dropped_bytes": dropped_bytes,
                "state_reason": reason,
            }

        poll_result = await asyncio.to_thread(
            session.poll_output,
            timeout=min(0.25, remaining),
            quiescence_ms=quiet_ms,
        )
        chunk = poll_result.get("output", "")
        if chunk:
            chunks.append(chunk)
        output_truncated = output_truncated or bool(poll_result.get("output_truncated", False))
        dropped_bytes += int(poll_result.get("dropped_bytes", 0))

        terminal_state, reason = await _describe_terminal(session, ctx)
        if terminal_state == "ready":
            return {
                "outcome": "success",
                "terminal_state": terminal_state,
                "output": _maybe_strip_ansi("".join(chunks), strip_ansi=strip_ansi),
                "prompt_detected": True,
                "output_truncated": output_truncated,
                "dropped_bytes": dropped_bytes,
                "state_reason": reason,
            }
        if poll_result.get("status") == "eof" or not session.alive:
            return {
                "outcome": "eof",
                "terminal_state": None,
                "output": _maybe_strip_ansi("".join(chunks), strip_ansi=strip_ansi),
                "prompt_detected": False,
                "output_truncated": output_truncated,
                "dropped_bytes": dropped_bytes,
                "state_reason": "pty eof",
            }
        if poll_result.get("status") == "error":
            return {
                "outcome": "error",
                "terminal_state": terminal_state,
                "output": _maybe_strip_ansi("".join(chunks), strip_ansi=strip_ansi),
                "prompt_detected": False,
                "output_truncated": output_truncated,
                "dropped_bytes": dropped_bytes,
                "state_reason": reason,
                "error": str(poll_result.get("error", "pty error")),
            }


@mcp.tool()
async def get_metadata(session_id: str, ctx: Context | None = None) -> dict:
    """Get metadata for an existing PTY session.

    Requires an existing session created via `create_session(session_id, cwd)`.
    """
    if session_id in getattr(session_manager, "_terminated", set()):
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "cwd": None,
            "pid": None,
            "cols": None,
            "rows": None,
            "started_at": None,
            "last_activity_at": None,
            "description": None,
            "shell_prompt_regex": None,
            "state_reason": "",
        }
    if session_id not in session_manager.sessions:
        return {
            "outcome": "invalid_session",
            "terminal_state": None,
            "cwd": None,
            "pid": None,
            "cols": None,
            "rows": None,
            "started_at": None,
            "last_activity_at": None,
            "description": None,
            "shell_prompt_regex": None,
            "state_reason": _missing_session_hint(session_id),
        }
    session = session_manager.sessions[session_id]

    terminal_state, reason = await _describe_terminal(session, ctx)
    meta = await asyncio.to_thread(session.metadata)
    out = {
        k: meta.get(k)
        for k in [
            "cwd",
            "pid",
            "cols",
            "rows",
            "started_at",
            "last_activity_at",
            "description",
            "shell_prompt_regex",
        ]
    }
    out.update({"state_reason": reason})
    return {"outcome": "success", "terminal_state": terminal_state, **out}


@mcp.tool()
def list_sessions() -> dict:
    return {"sessions": session_manager.list_sessions()}


@mcp.tool()
async def configure_session(
    session_id: str,
    description: Annotated[
        str | None,
        Field(
            description="Free-form description of what this session is doing (for humans). Does not affect execution."
        ),
    ] = None,
    shell_prompt_regex: Annotated[
        str | None,
        Field(
            description=(
                "Optional regex used to detect when the terminal is idle at a shell prompt. "
                "Matched against the last visible non-empty screen line. Prefer anchoring to the end."
            )
        ),
    ] = None,
    ctx: Context | None = None,
) -> dict:
    """Update session metadata and shell prompt detection.

    `description` is free-form metadata for humans and does not affect execution.

    `shell_prompt_regex` is used by heuristic readiness detection to decide
    whether the terminal is idle at a shell prompt. It is matched against the
    last non-empty line of the rendered screen.

    This can be called before a session exists; values are stored by `session_id`
    and applied when the session is created.
    """
    if session_id in getattr(session_manager, "_terminated", set()):
        return {"outcome": "terminated", "terminal_state": None, "state_reason": ""}

    try:
        session_manager.configure_full(
            session_id,
            description=description,
            shell_prompt_regex=shell_prompt_regex,
        )
    except RuntimeError as e:
        if str(e) == "terminated":
            return {"outcome": "terminated", "terminal_state": None, "state_reason": ""}
        raise ValueError(str(e))

    session = session_manager.sessions.get(session_id)
    if session is None:
        cfg = session_manager._config.get(session_id, {})
        return {
            "outcome": "success",
            "terminal_state": None,
            "session_exists": False,
            "description": cfg.get("description"),
            "shell_prompt_regex": cfg.get("shell_prompt_regex"),
            "state_reason": "configured (session not yet created; call create_session(session_id, cwd))",
        }

    terminal_state, reason = await _describe_terminal(session, ctx)
    return {
        "outcome": "success",
        "terminal_state": terminal_state,
        "session_exists": True,
        "description": getattr(session, "description", None),
        "shell_prompt_regex": getattr(session, "shell_prompt_regex", None),
        "state_reason": reason,
    }


@mcp.tool()
async def send_signal(
    session_id: str,
    signal: str,
    deadline_s: float = 5.0,
    quiet_ms: int = QUIESCENCE_MS,
    strip_ansi: bool = True,
    ctx: Context | None = None,
) -> dict:
    """Send an OS signal to the foreground process in a session.

    Requires an existing session created via `create_session(session_id, cwd)`.
    """
    if session_id in getattr(session_manager, "_terminated", set()):
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": "",
        }
    if session_id not in session_manager.sessions:
        return {
            "outcome": "invalid_session",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": _missing_session_hint(session_id),
        }
    try:
        session = session_manager.get_session(session_id)
    except RuntimeError:
        return {
            "outcome": "terminated",
            "terminal_state": None,
            "output": "",
            "output_truncated": False,
            "dropped_bytes": 0,
            "state_reason": "",
        }

    signum = None
    s = str(signal).strip()
    if s.isdigit():
        signum = int(s)
    else:
        name = s.upper()
        if not name.startswith("SIG"):
            name = "SIG" + name
        signum = getattr(signal_mod, name, None)
        if signum is None:
            raise ValueError(f"Unknown signal: {signal!r}")
        signum = int(signum)

    deadline_s = _clamp_tool_timeout(deadline_s)
    result = await asyncio.to_thread(session.send_signal, signum, deadline_s, True, quiet_ms)
    return await _build_stream_response(session=session, ctx=ctx, result=result, strip_ansi=strip_ansi)


@mcp.tool()
def transcript(session_id: str) -> dict:
    """Get the transcript file path for this session."""
    if session_id in getattr(session_manager, "_terminated", set()):
        return {"outcome": "terminated", "terminal_state": None, "transcript": None, "state_reason": ""}
    if session_id not in session_manager.sessions:
        hint = _missing_session_hint(session_id)
        tp = _session_transcript_path_if_exists(session_id)
        if tp:
            return {"outcome": "invalid_session", "terminal_state": None, "transcript": tp, "state_reason": hint}
        return {"outcome": "invalid_session", "terminal_state": None, "transcript": None, "state_reason": hint}
    session = session_manager.sessions[session_id]

    snap = session.screen_snapshot(log=False, drain=False)
    state, reason = detect_state_heuristic(
        snap["screen"],
        cursor_x=snap.get("cursor_x"),
        shell_prompt_regex=getattr(session, "shell_prompt_regex", None),
    )
    return {
        "outcome": "success",
        "terminal_state": _terminal_state_from_state(state) if session.alive else None,
        "transcript": session.transcript(),
        "state_reason": reason if session.alive else "pty eof",
    }


@mcp.tool()
async def terminate(session_id: str) -> dict:
    """Terminate a PTY session. Future calls using the same `session_id` are rejected."""
    session_manager._terminated.add(session_id)
    if session_id in session_manager.sessions:
        await asyncio.to_thread(session_manager.sessions[session_id].terminate)
        del session_manager.sessions[session_id]
        session_manager._last_used.pop(session_id, None)
    return {"outcome": "terminated", "terminal_state": None, "state_reason": ""}


def signal_handler(sig, frame):
    """Handle signals for graceful shutdown."""
    session_manager.terminate_all()
    sys.exit(0)


def main():
    """Main entry point for MCP server."""
    _configure_logging()
    signal_mod.signal(signal_mod.SIGINT, signal_handler)
    signal_mod.signal(signal_mod.SIGTERM, signal_handler)
    mcp.run()


if __name__ == "__main__":
    main()
