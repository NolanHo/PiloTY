import asyncio

from piloty import mcp_server


def test_mcp_tool_shapes_include_status_and_prompt(tmp_path):
    prev = mcp_server.QUIESCENCE_MS
    mcp_server.QUIESCENCE_MS = 50
    session_id = "test_mcp_shapes"
    try:
        created = asyncio.run(mcp_server.create_session(session_id=session_id, cwd=str(tmp_path)))
        assert created["created"] is True

        r = asyncio.run(mcp_server.run(session_id=session_id, command="echo hi", timeout=2.0))
        assert "screen" not in r
        assert set(r.keys()) >= {"status", "prompt", "output", "timed_out"}

        r = asyncio.run(mcp_server.send_input(session_id=session_id, text="echo si\n", timeout=2.0))
        assert "screen" not in r
        assert set(r.keys()) >= {"status", "prompt", "output", "timed_out"}

        r = asyncio.run(mcp_server.send_control(session_id=session_id, key="l", timeout=2.0))
        assert "screen" not in r
        assert set(r.keys()) >= {"status", "prompt", "output", "timed_out"}

        r = asyncio.run(mcp_server.poll_output(session_id=session_id, timeout=0.05))
        assert set(r.keys()) >= {"status", "prompt", "output", "timed_out"}

        r = asyncio.run(mcp_server.send_password(session_id=session_id, password="not_a_secret", timeout=2.0))
        assert "screen" not in r
        assert set(r.keys()) >= {"status", "prompt", "output", "timed_out"}
        assert r["output"].startswith("[password sent]")
        assert "not_a_secret" not in r["output"]

        s = asyncio.run(mcp_server.get_screen(session_id=session_id))
        assert set(s.keys()) >= {"status", "prompt", "screen"}
    finally:
        try:
            asyncio.run(mcp_server.terminate(session_id))
        except Exception:
            pass
        mcp_server.QUIESCENCE_MS = prev


def test_poll_output_timeout_is_capped(tmp_path):
    session_id = "test_poll_output_timeout_cap"
    try:
        asyncio.run(mcp_server.create_session(session_id=session_id, cwd=str(tmp_path)))
        session = mcp_server.session_manager.sessions[session_id]

        captured: dict[str, float] = {}

        def patched_poll_output(*args, **kwargs):
            captured["timeout"] = float(kwargs.get("timeout"))
            return {"status": "timeout", "output": "", "output_truncated": False, "dropped_bytes": 0}

        session.poll_output = patched_poll_output  # type: ignore[method-assign]

        asyncio.run(mcp_server.poll_output(session_id=session_id, timeout=999999.0))
        assert captured["timeout"] == mcp_server.MAX_TOOL_TIMEOUT_S
    finally:
        try:
            asyncio.run(mcp_server.terminate(session_id))
        except Exception:
            pass


def test_public_tool_timeouts_are_capped(tmp_path):
    session_id = "test_public_tool_timeout_cap"
    try:
        asyncio.run(mcp_server.create_session(session_id=session_id, cwd=str(tmp_path)))
        session = mcp_server.session_manager.sessions[session_id]

        captured: list[float] = []

        def patched_type(*args, **kwargs):
            captured.append(float(kwargs.get("timeout")))
            return {"status": "timeout", "output": "", "output_truncated": False, "dropped_bytes": 0}

        def patched_expect(pattern, timeout, log=True):
            captured.append(float(timeout))
            return {"status": "timeout", "output": "", "output_truncated": False, "dropped_bytes": 0, "match": None, "groups": []}

        session.type = patched_type  # type: ignore[method-assign]
        session.expect = patched_expect  # type: ignore[method-assign]

        asyncio.run(mcp_server.run(session_id=session_id, command="echo hi", timeout=999999.0))
        asyncio.run(mcp_server.send_input(session_id=session_id, text="echo hi\n", timeout=999999.0))
        asyncio.run(mcp_server.send_password(session_id=session_id, password="not_a_secret", timeout=999999.0))
        asyncio.run(mcp_server.send_control(session_id=session_id, key="c", timeout=999999.0))
        asyncio.run(mcp_server.expect(session_id=session_id, pattern="hi", timeout=999999.0))

        assert captured == [mcp_server.MAX_TOOL_TIMEOUT_S] * 5
    finally:
        try:
            asyncio.run(mcp_server.terminate(session_id))
        except Exception:
            pass


def test_mcp_terminate_is_final(tmp_path):
    session_id = "test_mcp_terminate_final"
    try:
        asyncio.run(mcp_server.create_session(session_id=session_id, cwd=str(tmp_path)))
        asyncio.run(mcp_server.run(session_id=session_id, command="echo hi", timeout=2.0))
        asyncio.run(mcp_server.terminate(session_id))
        r = asyncio.run(mcp_server.run(session_id=session_id, command="echo nope", timeout=2.0))
        assert r["status"] == "terminated"
    finally:
        try:
            asyncio.run(mcp_server.terminate(session_id))
        except Exception:
            pass
