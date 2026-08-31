# -*- coding: utf-8 -*-
"""The earcon gateway: a transparent OpenAI-compatible proxy that learns.

Request path (every turn):
    client -> [experience injection] -> transparent forward -> upstream
    response streams back while the turn is recorded into a session buffer.

Learning path (session close, background thread):
    an LLM judge distills the trajectory into experience cards ->
    SQLite memory -> retrieved for future sessions' injections.

For free-text coding/chat sessions (no enumerable action space) the
JitRL logit adjustment is realized as context injection instead: it is
the same loop (experience -> behavior change) at the fidelity the
protocol allows. The core library (earcon.core) implements the discrete
logit path used by examples/treasure_maze.
"""

import json
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

DEFAULTS = {
    "session_timeout": 1800.0,   # seconds of inactivity -> session close
    "sweep_interval": 60.0,      # how often the reaper checks for timeouts
    "max_turns_judged": 40,
    "min_turns_to_judge": 3,
    "task_sig_chars": 120,       # task signature length for retrieval
    "inject_top_k": 5,           # cards per injection
    "inject_threshold": 0.15,    # |G| beyond which a card is useful/a lesson
    "retrieve_scan_limit": 3000, # recent-card scan window
    "retrieve_min_overlap": 2,   # keyword overlap to count as related
    "turn_user_chars": 2000,
    "turn_assistant_chars": 4000,
}

JUDGE_PROMPT = """Below is the full trajectory of a coding-assistant session
(user request -> assistant key response). Distill 1-5 reusable experience
cards from the assistant's performance. One card per line, format:
decision=<a key decision the assistant got right or wrong, max 30 words> ; score=<integer -3..3, positive if right, negative if wrong>
Output only those lines.

{steps}"""


class GatewayMemory:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS cards(
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   task TEXT,
                   action TEXT,
                   G REAL,
                   created REAL)""")
        self.conn.commit()

    def add(self, task, action, G):
        with self.lock:
            self.conn.execute(
                "INSERT INTO cards(task, action, G, created) VALUES(?,?,?,?)",
                (task, action, G, time.time()))
            self.conn.commit()

    def size(self):
        return self.conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]

    def retrieve_top(self, task_sig, scan_limit, min_overlap, k):
        rows = self.conn.execute(
            "SELECT task, action, G FROM cards ORDER BY id DESC LIMIT ?",
            (scan_limit,)).fetchall()
        if not rows:
            return []
        q = set(re.findall(r"[\w一-鿿]+", (task_sig or "").lower()))
        scored = []
        for task, action, G in rows:
            t = set(re.findall(r"[\w一-鿿]+", (task or "").lower()))
            inter = len(q & t)
            if inter >= min_overlap:
                scored.append((inter, G, action))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return scored[:k]


class Gateway:
    def __init__(self, upstream, api_key, db_path, judge_model,
                 inject=True, extra_body=None, config=None):
        self.upstream = upstream.rstrip("/")
        self.api_key = api_key
        self.judge_model = judge_model
        self.inject = inject
        self.extra_body = dict(extra_body or {})
        self.conf = dict(DEFAULTS)
        self.conf.update(config or {})
        self.mem = GatewayMemory(db_path)

        self.sessions = {}            # sid -> {"turns": [...], "last": ts}
        self.session_lock = threading.Lock()
        self._stop = threading.Event()
        self.reaper = threading.Thread(target=self._reap_loop, daemon=True)
        self.reaper.start()

    # ---------------- session bookkeeping ----------------

    def record_turn(self, sid, user_text, assistant_text):
        c = self.conf
        with self.session_lock:
            s = self.sessions.setdefault(sid, {"turns": [], "last": time.time()})
            s["turns"].append((user_text[:c["turn_user_chars"]],
                               (assistant_text or "")[:c["turn_assistant_chars"]]))
            s["last"] = time.time()

    def close_session(self, sid):
        with self.session_lock:
            s = self.sessions.pop(sid, None)
        if s and len(s["turns"]) >= self.conf["min_turns_to_judge"]:
            threading.Thread(target=self._judge_and_write,
                             args=(s["turns"],), daemon=True).start()
        return len((s or {}).get("turns", []))

    def _reap_loop(self):
        while not self._stop.wait(self.conf["sweep_interval"]):
            now = time.time()
            stale = []
            with self.session_lock:
                for sid, s in list(self.sessions.items()):
                    if now - s["last"] > self.conf["session_timeout"]:
                        stale.append(sid)
            for sid in stale:
                self.close_session(sid)

    # ---------------- judging ----------------

    def _judge_and_write(self, turns):
        try:
            sig = self._task_signature(turns)
            if not sig:
                return
            c = self.conf
            lines = ["%d. user: %s" % (i + 1, u[:600])
                     for i, (u, a) in enumerate(turns[:c["max_turns_judged"]])]
            steps = "\n".join(
                "%d. user: %s\n   assistant: %s"
                % (i + 1, u[:600], a[:600])
                for i, (u, a) in enumerate(turns[:c["max_turns_judged"]]))
            text = self._llm_chat(JUDGE_PROMPT.format(steps=steps))
            for line in text.splitlines():
                m = re.search(r"decision=(.+?)\s*;\s*score=(-?\d+)", line)
                if m:
                    self.mem.add(sig, m.group(1).strip(),
                                 int(m.group(2)) / 3.0)
        except Exception as e:
            print("[earcon] judge error:", e)

    def _task_signature(self, turns):
        for u, _ in turns:
            if u.strip():
                return u[:self.conf["task_sig_chars"]]
        return ""

    def _llm_chat(self, prompt, max_tokens=2000):
        payload = {"model": self.judge_model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "stream": False,
                   "temperature": 0.0}
        payload.update(self.extra_body)
        req = urllib.request.Request(
            self.upstream + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + (self.api_key or "")})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read().decode("utf-8"))
        return d["choices"][0]["message"].get("content") or ""

    # ---------------- injection ----------------

    def experience_prefix(self, task_sig):
        c = self.conf
        top = self.mem.retrieve_top(task_sig, c["retrieve_scan_limit"],
                                    c["retrieve_min_overlap"], c["inject_top_k"])
        if not top:
            return None
        lines = ["[earcon experience] lessons from past similar sessions:"]
        for _, G, action in top:
            tag = ("worked" if G > c["inject_threshold"]
                   else "failed" if G < -c["inject_threshold"] else "mixed")
            lines.append("- (%s) %s" % (tag, action))
        lines.append("Treat these as historical summaries; use your own "
                     "judgment for the current task.")
        return "\n".join(lines)

    # ---------------- HTTP ----------------

    def _forward(self, body):
        req = urllib.request.Request(
            self.upstream + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + (self.api_key or "")})
        return urllib.request.urlopen(req, timeout=300)

    def handle_chat(self, body, auth_key):
        messages = body.get("messages", [])

        # session id: honor explicit metadata, else one request per session
        # is naturally grouped by close_session/timeout sweeps
        sid = (body.get("metadata") or {}).get("session_id") or "default"

        if self.inject and messages:
            sig = (_first_user_text(messages) or "")[:self.conf["task_sig_chars"]]
            if sig.strip():
                prefix = self.experience_prefix(sig)
                if prefix:
                    new_msgs, inserted = [], False
                    for m in messages:
                        new_msgs.append(m)
                        if not inserted and m.get("role") == "system":
                            new_msgs.append({"role": "system", "content": prefix})
                            inserted = True
                    if not inserted:
                        new_msgs.insert(0, {"role": "system", "content": prefix})
                    body = dict(body)
                    body["messages"] = new_msgs

        upstream_key = self.api_key or auth_key
        stream = body.get("stream", False)
        body_json = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json",
                   "Authorization": "Bearer " + upstream_key}
        req = urllib.request.Request(self.upstream + "/chat/completions",
                                     data=body_json, headers=headers)
        try:
            up_resp = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as e:
            # surface upstream errors (4xx/5xx bodies) instead of a bare 500
            detail = e.read().decode("utf-8", "replace")[:2000]
            return {"error": {"upstream_status": e.code,
                              "upstream_body": detail}}
        except (urllib.error.URLError, OSError) as e:
            return {"error": {"upstream_status": 502, "upstream_body": str(e)}}

        if stream:
            def gen():
                collected = []
                try:
                    for chunk in up_resp:
                        yield chunk
                        try:
                            line = chunk.decode("utf-8").strip()
                            if line.startswith("data: ") and line != "data: [DONE]":
                                j = json.loads(line[6:])
                                delta = (j.get("choices") or [{}])[0].get("delta", {})
                                if delta.get("content"):
                                    collected.append(delta["content"])
                        except Exception:
                            pass
                finally:
                    if collected:
                        self.record_turn(sid, _first_user_text(messages) or "",
                                         "".join(collected))

            return StreamingResponse(gen(), media_type="text/event-stream")
        else:
            d = json.loads(up_resp.read().decode("utf-8"))
            content = (d["choices"][0]["message"].get("content") or "")
            self.record_turn(sid, _first_user_text(messages) or "", content)
            return d


def _first_user_text(messages):
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
    return ""


def create_app(gateway):
    app = FastAPI()

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def proxy_chat(request: Request):
        body = await request.json()
        auth_key = request.headers.get("Authorization", "").replace("Bearer ", "")
        result = gateway.handle_chat(body, auth_key)
        if isinstance(result, StreamingResponse):
            return result
        return JSONResponse(result)

    @app.get("/v1/models")
    @app.get("/models")
    async def models():
        req = urllib.request.Request(
            gateway.upstream + "/models",
            headers={"Authorization": "Bearer " + (gateway.api_key or "")})
        with urllib.request.urlopen(req, timeout=30) as r:
            return JSONResponse(json.loads(r.read().decode("utf-8")))

    @app.get("/v1/earcon/stats")
    async def stats():
        return JSONResponse({"cards": gateway.mem.size(),
                             "sessions_open": len(gateway.sessions),
                             "inject": gateway.inject})

    @app.post("/v1/earcon/session/{sid}/close")
    async def close_session(sid: str):
        n = gateway.close_session(sid)
        return JSONResponse({"closed": sid, "turns": n})

    return app
