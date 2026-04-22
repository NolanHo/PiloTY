import shutil

from mcp.server.fastmcp.utilities.context_injection import find_context_parameter

from piloty import mcp_server
from piloty import core
from piloty.core import PTY


def test_fastmcp_context_injection_detects_ctx_param():
    assert find_context_parameter(mcp_server.send_line) == "ctx"
    assert find_context_parameter(mcp_server.wait_for_output) == "ctx"


def test_basic_pty_command():
    pty = PTY(session_id="test_basic")
    try:
        r = pty.type("echo hello\n", timeout=5.0, quiescence_ms=300)
        assert r["status"] == "quiescent"
        assert "hello" in r["output"]
    finally:
        pty.terminate()


def test_vt100_private_csi_does_not_crash():
    pty = PTY(session_id="test_private_csi")
    try:
        r = pty.type("printf '\\033[?1m\\033[0m\\n'\n", timeout=5.0)
        assert r["status"] == "quiescent"
        r = pty.type("printf '\\033[?25l\\033[?25h\\n'\n", timeout=5.0)
        assert r["status"] == "quiescent"
        assert getattr(pty, "_vt100_error", None) is None
    finally:
        pty.terminate()


def test_transcript_written():
    pty = PTY(session_id="test_transcript")
    try:
        path = pty.transcript()
        r = pty.type("echo 'transcript test'\n", timeout=5.0)
        assert r["status"] == "quiescent"
        assert "transcript test" in open(path, "r", encoding="utf-8").read()
    finally:
        pty.terminate()


def test_ssh_version_does_not_crash_when_present():
    if shutil.which("ssh") is None:
        return
    pty = PTY(session_id="test_ssh_version")
    try:
        r = pty.type("ssh -V\n", timeout=5.0)
        assert r["status"] in ("quiescent", "timeout")
    finally:
        pty.terminate()


def test_pty_public_timeouts_are_capped(monkeypatch):
    pty = PTY(session_id="test_pty_timeout_cap")
    try:
        captured: list[float] = []

        def patched_drain(*, quiescence_ms=None, timeout=30.0, log=True, capture=True, require_output=False):
            captured.append(float(timeout))
            return "timeout"

        pty._drain = patched_drain  # type: ignore[method-assign]
        pty.type("echo hello\n", timeout=999999.0, quiescence_ms=300)
        pty.poll_output(timeout=999999.0, quiescence_ms=300)
        pty.send_signal(0, timeout=999999.0)
        assert captured == [core.MAX_TIMEOUT_S] * 3
    finally:
        pty.terminate()


def test_pty_expect_timeout_is_capped(monkeypatch):
    pty = PTY(session_id="test_pty_expect_timeout_cap")
    try:
        ticks = iter([0.0, 301.0])
        monkeypatch.setattr(core.time, "monotonic", lambda: next(ticks))

        def patched_read_nonblocking(*args, **kwargs):
            raise AssertionError("expect should time out before reading")

        monkeypatch.setattr(pty._process, "read_nonblocking", patched_read_nonblocking)

        result = pty.expect("needle", timeout=999999.0, log=False)
        assert result["status"] == "timeout"
    finally:
        pty.terminate()
