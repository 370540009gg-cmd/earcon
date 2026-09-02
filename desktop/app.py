# -*- coding: utf-8 -*-
"""earcon 桌面客户端（macOS 优先）。

一个原生窗口（pywebview + 系统 WebKit），内嵌 earcon 的 Web UI；
启动时自动拉起本地网关子进程（如果没在跑），退出时收尾。

用法（示意级初版）：
    python desktop/app.py            # 连接/拉起默认网关 (127.0.0.1:8800)
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

import webview

GATEWAY = "http://127.0.0.1:8800"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HTML = os.path.join(HERE, "ui.html")


def gateway_alive():
    try:
        with urllib.request.urlopen(GATEWAY + "/v1/earcon/stats", timeout=1.5):
            return True
    except Exception:
        return False


def start_gateway():
    """拉起网关子进程（演示配置：读 zcode 路由 + 本地 judge 兜底）。"""
    log = open(os.path.join(REPO, "gateway.log"), "a")
    p = subprocess.Popen(
        [sys.executable, "-m", "earcon.cli", "serve",
         "--judge-upstream", os.environ.get("EARCON_JUDGE_UPSTREAM",
                                            "http://llm-gw.jd.local/v1"),
         "--judge-api-key", os.environ.get("EARCON_JUDGE_API_KEY", ""),
         "--judge-model", os.environ.get("EARCON_JUDGE_MODEL",
                                         "GLM-5.2-joybuilder"),
         "--routes-from", "zcode",
         "--port", "8800",
         "--db", os.path.join(REPO, "earcon_memory.db")],
        cwd=REPO, stdout=log, stderr=log,
        env=dict(os.environ, PYTHONPATH=REPO))
    # 等它就绪
    for _ in range(40):
        if gateway_alive():
            return p
        time.sleep(0.25)
    return p


class Api:
    """暴露给 JS 的本地接口：读网关数据 + 关闭 App。"""

    def __init__(self, gw_proc):
        self.gw_proc = gw_proc

    def fetch_stats(self):
        return _get("/v1/earcon/ui/stats") or {}

    def fetch_cards(self, offset=0, limit=50, q="", tag=""):
        return _get("/v1/earcon/ui/cards?offset=%s&limit=%s&q=%s&tag=%s"
                    % (offset, limit, q, tag)) or {"total": 0, "cards": []}

    def fetch_sessions(self):
        return _get("/v1/earcon/ui/sessions") or {}

    def delete_card(self, card_id):
        try:
            req = urllib.request.Request(
                "%s/v1/earcon/ui/cards/%s" % (GATEWAY, card_id),
                method="DELETE")
            with urllib.request.urlopen(req, timeout=3) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            return {}

    def quit_app(self):
        window.destroy()


def _get(path):
    try:
        with urllib.request.urlopen(GATEWAY + path, timeout=3) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        import sys
        print("[earcon bridge error]", path, repr(e), file=sys.stderr)
        return None


if __name__ == "__main__":
    proc = None
    if not gateway_alive():
        proc = start_gateway()
        print("网关已由客户端拉起" if gateway_alive() else "网关启动超时，UI 将以离线模式运行")

    api = Api(proc)
    # load local HTML file, use pywebview bridge for data
    window = webview.create_window(
        "earcon", HTML, js_api=api, width=1080, height=760,
        min_size=(860, 600),
        text_select=False)
    try:
        webview.start(debug=True)
    finally:
        # gateway keeps running after the app closes - sessions survive
        # (user manages the gateway lifecycle separately)
        pass
