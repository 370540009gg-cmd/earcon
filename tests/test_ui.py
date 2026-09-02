# -*- coding: utf-8 -*-
"""UI API tests: stats, card list/search/delete, sessions view, static mount."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from earcon.gateway import Gateway, create_app


class MockUpstream(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        resp = {"choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}]}
        out = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture()
def stack(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), MockUpstream)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    gw = Gateway(judge_model="m", db_path=str(tmp_path / "g.db"),
                 judge_upstream="http://127.0.0.1:1/v1", judge_api_key="j",
                 routes=["m-test=http://127.0.0.1:%d/v1" % port],
                 config={"session_timeout": 0.2, "sweep_interval": 0.1})
    yield gw, port
    gw._stop.set()
    server.shutdown()


def _client(gw):
    from fastapi.testclient import TestClient
    return TestClient(create_app(gw))


def _seed_cards(gw):
    gw.mem.add("在 zcode 配置 provider", "先备份再改 config", 0.9)
    gw.mem.add("排查连接池报错", "先看监控再查代码", 0.6)
    gw.mem.add("改告警规则", "没确认 ID 直接改错了", -0.8)
    gw.mem.add("写周报", "从 git log 生成草稿", 0.05)  # pending
    return gw


def test_ui_stats_summary(stack):
    gw, _ = _seed_cards(stack[0]), None
    with _client(stack[0]) as c:
        r = c.get("/v1/earcon/ui/stats")
    d = r.json()
    assert d["total"] == 4 and d["positive"] == 2 and d["negative"] == 1
    assert d["pending"] == 1 and d["today"] == 4
    assert sum(d["histogram"]) == 4 and len(d["daily"]) == 7


def test_ui_cards_list_search_filter(stack):
    _seed_cards(stack[0])
    with _client(stack[0]) as c:
        all_r = c.get("/v1/earcon/ui/cards").json()
        assert all_r["total"] == 4 and len(all_r["cards"]) == 4
        neg = c.get("/v1/earcon/ui/cards?tag=negative").json()
        assert neg["total"] == 1 and "没确认 ID" in neg["cards"][0]["action"]
        hit = c.get("/v1/earcon/ui/cards?q=连接池").json()
        assert hit["total"] == 1
        miss = c.get("/v1/earcon/ui/cards?q=不存在的关键词").json()
        assert miss["total"] == 0


def test_ui_card_delete(stack):
    _seed_cards(stack[0])
    with _client(stack[0]) as c:
        cards = c.get("/v1/earcon/ui/cards").json()["cards"]
        rid = cards[0]["id"]
        r = c.delete("/v1/earcon/ui/cards/%d" % rid)
        assert r.json()["deleted"] is True
        assert c.get("/v1/earcon/ui/cards").json()["total"] == 3
        r2 = c.delete("/v1/earcon/ui/cards/99999")
        assert r2.json()["deleted"] is False


def test_ui_sessions_view(stack):
    gw, _ = stack
    gw.record_turn("demo-sid", "用户：帮我看报错", "助手：先看监控")
    gw.record_turn("demo-sid", "用户：然后呢", "助手：调大连接数")
    with _client(gw) as c:
        live = c.get("/v1/earcon/ui/sessions").json()["live"]
        assert live[0]["sid"] == "demo-sid" and live[0]["turns"] == 2
    # 关闭后应出现在已关闭历史（带卡片数）
    gw.close_session("demo-sid", reason="manual")
    with _client(gw) as c:
        closed = c.get("/v1/earcon/ui/sessions").json()["closed"]
        assert closed and closed[0]["sid"] == "demo-sid"
        assert closed[0]["turns"] == 2


def test_ui_static_page_removed(stack):
    """/ui/ was intentionally removed - the desktop client loads the local
    HTML directly (file:// + CORS), no need for the gateway to serve it."""
    with _client(stack[0]) as c:
        r = c.get("/ui/")
    assert r.status_code == 404
