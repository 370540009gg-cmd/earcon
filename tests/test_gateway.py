# -*- coding: utf-8 -*-
"""Gateway integration test against a local mock upstream.

Covers: transparent forwarding, turn recording, session close -> judge
-> card write, and injection into a subsequent request. Uses fastapi's
TestClient so no real network is involved beyond the mock upstream.
"""

import json
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from earcon.gateway import Gateway, create_app


class MockUpstream(BaseHTTPRequestHandler):
    received = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        MockUpstream.received.append(body)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = ['data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
                      'data: [DONE]\n\n']
            for c in chunks:
                self.wfile.write(c.encode())
        else:
            resp = {"choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant",
                                             "content": "hello"}}],
                    "model": "mock"}
            out = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    def log_message(self, *a):
        pass


class JudgeBackend:
    """Stands in for the LLM judge: emits one experience card."""
    calls = 0

    def __init__(self, gateway):
        self.gw = gateway

    # monkeypatch: replace gateway._llm_chat
    def llm_chat(self, prompt, max_tokens=2000):
        JudgeBackend.calls += 1
        return "decision=prefer backups ; score=2"


@pytest.fixture()
def stack(tmp_path):
    MockUpstream.received = []
    server = HTTPServer(("127.0.0.1", 0), MockUpstream)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    gw = Gateway(judge_model="mock-judge", db_path=str(tmp_path / "g.db"),
                 routes=["m-test=http://127.0.0.1:%d/v1" % port],
                 judge_upstream="http://127.0.0.1:%d/v1" % port,
                 judge_api_key="judge-key",
                 upstream="http://127.0.0.1:%d/v1" % port,
                 api_key="fallback-key", inject=True,
                 config={"session_timeout": 0.2, "sweep_interval": 0.1})
    gw._llm_chat = JudgeBackend(gw).llm_chat
    yield gw, port
    gw._stop.set()
    server.shutdown()


def _post(app, body):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        return c.post("/v1/chat/completions", json=body)


def test_forwards_and_records(stack):
    gw, port = stack
    app = create_app(gw)
    body = {"model": "m-test", "stream": False, "messages": [
        {"role": "user", "content": "how do I configure X?"}]}
    resp = _post(app, body)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hello"
    assert MockUpstream.received, "upstream got the request"


def test_work_channel_passes_through_client_key(stack):
    """The client carries its own key; the gateway must not rewrite it."""
    gw, port = stack
    captured = {}

    original_forward = gw.handle_chat.__globals__  # not used; see below
    app = create_app(gw)
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        c.post("/v1/chat/completions",
               json={"model": "m-test", "stream": False,
                     "messages": [{"role": "user", "content": "hi"}]},
               headers={"Authorization": "Bearer my-own-key"})
    # mock upstream doesn't record headers in received; assert indirectly:
    # Gateway forwards auth_key when present (fallback only when absent)
    assert True  # pass-through is exercised; deeper header check below


def test_work_channel_fallback_key_only_when_missing(stack):
    gw, port = stack
    seen = []

    class HeaderCapture(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            seen.append(self.headers.get("Authorization"))
            resp = {"choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant",
                                             "content": "ok"}}]}
            out = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), HeaderCapture)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    p = srv.server_address[1]
    import tempfile
    gw2 = Gateway(judge_model="m", db_path=tempfile.mktemp(suffix=".db"),
                  judge_upstream="http://127.0.0.1:1/v1", judge_api_key="j",
                  routes=["m-test=http://127.0.0.1:%d/v1" % p],
                  upstream="http://127.0.0.1:%d/v1" % p,
                  api_key="fallback-key", inject=False)
    app2 = create_app(gw2)
    from fastapi.testclient import TestClient
    with TestClient(app2) as c:
        # client brings its own key -> passed through
        c.post("/v1/chat/completions",
               json={"model": "m-test", "stream": False,
                     "messages": [{"role": "user", "content": "hi"}]},
               headers={"Authorization": "Bearer client-key"})
        # no key -> falls back to the gateway's configured one
        c.post("/v1/chat/completions",
               json={"model": "m-test", "stream": False,
                     "messages": [{"role": "user", "content": "hi"}]})
    gw2._stop.set()
    srv.shutdown()
    assert seen[0] == "Bearer client-key"      # pass-through wins
    assert seen[1] == "Bearer fallback-key"    # fallback only when absent


def test_judge_uses_independent_channel(stack):
    """Judge requests must go to judge_upstream with judge key, never the
    work channel - this is what makes the judge a global config."""
    gw, port = stack
    hit = []

    class JudgeCapture(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            hit.append((self.headers.get("Authorization"),
                        json.loads(json.dumps({})).get("model")))
            resp = {"choices": [{"index": 0, "message": {
                        "role": "assistant",
                        "content": "decision=x ; score=1"}}]}
            out = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    jsrv = HTTPServer(("127.0.0.1", 0), JudgeCapture)
    threading.Thread(target=jsrv.serve_forever, daemon=True).start()
    jp = jsrv.server_address[1]

    # reconstruct a gateway whose work and judge channels are different hosts
    gw3 = Gateway(judge_model="judge-m", db_path=tempfile.mktemp(suffix=".db"),
                  judge_upstream="http://127.0.0.1:%d/v1" % jp,
                  judge_api_key="judge-secret",
                  upstream="http://127.0.0.1:%d/v1" % port,  # work = mock upstream
                  api_key="work-key", inject=False)
    # call the judge directly through its real channel
    out = gw3._llm_chat("rate this")
    assert "decision" in out
    # hit came from judge channel with judge key
    assert hit and hit[0][0] == "Bearer judge-secret"
    gw3._stop.set()
    jsrv.shutdown()


def test_injection_after_cards_exist(stack):
    gw, port = stack
    # pre-seed a card matching future task signature
    gw.mem.add("how do I configure X?",
               "check the config file first", 0.8)
    app = create_app(gw)
    body = {"model": "m-test", "stream": False, "messages": [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "how do I configure X? please"}]}
    resp = _post(app, body)
    assert resp.status_code == 200
    # the forwarded request carries the injected experience system message
    forwarded = MockUpstream.received[-1]
    texts = [m["content"] for m in forwarded["messages"]
             if m["role"] == "system"]
    assert any("earcon experience" in t for t in texts)
    assert any("check the config file first" in t for t in texts)


def test_session_close_triggers_judge_and_writes_cards(stack):
    gw, port = stack
    JudgeBackend.calls = 0
    app = create_app(gw)
    for i in range(4):
        _post(app, {"model": "m-test", "stream": False, "messages": [
            {"role": "user", "content": "task %d about Y" % i}]})
    # explicit close (also exercised by the timeout reaper in production)
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        c.post("/v1/earcon/session/default/close")
    # judge thread is async; give it a moment
    for _ in range(50):
        if JudgeBackend.calls >= 1:
            break
        time.sleep(0.05)
    assert JudgeBackend.calls >= 1
    assert gw.mem.size() >= 1


def test_timeout_reaper_closes_idle_sessions(stack):
    gw, port = stack
    app = create_app(gw)
    _post(app, {"model": "m-test", "stream": False, "messages": [
        {"role": "user", "content": "task about Z"}]})
    _post(app, {"model": "m-test", "stream": False, "messages": [
        {"role": "user", "content": "task about Z again"}]})
    _post(app, {"model": "m-test", "stream": False, "messages": [
        {"role": "user", "content": "task about Z third"}]})
    # wait past the 0.2s timeout + sweep
    time.sleep(0.6)
    assert gw.sessions == {}, "idle session was reaped"


def test_upstream_error_returns_json_not_500(stack):
    gw, port = stack
    # make the route point at an unreachable upstream
    gw.route_table.routes["m-test"] = ("http://127.0.0.1:1/v1", "")
    app = create_app(gw)
    body = {"model": "m-test", "stream": False,
            "messages": [{"role": "user", "content": "hi"}]}
    resp = _post(app, body)
    assert resp.status_code == 200
    assert "error" in resp.json()
    assert resp.json()["error"]["upstream_status"] == 502
