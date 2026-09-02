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
import os
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

JUDGE_PROMPT = """以下是编程助手一次会话的完整轨迹（用户提问 → 助手关键回答）。
请从助手的表现中提炼 1-5 张可复用的经验卡片，每行一张，格式：
decision=<助手做对或做错的关键决策，30字内，用中文> ; score=<-3..3 的整数，做对为正，做错为负>
只输出这些行，不要解释。

{steps}"""


class GatewayMemory:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS cards(
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   task TEXT,
                   title TEXT,
                   action TEXT,
                   G REAL,
                   created REAL)""")
        # migrate: add title column to existing tables
        try:
            self.conn.execute("SELECT title FROM cards LIMIT 1")
        except sqlite3.OperationalError:
            self.conn.execute("ALTER TABLE cards ADD COLUMN title TEXT")
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS sessions(
                   sid TEXT PRIMARY KEY,
                   title TEXT,
                   opener TEXT,
                   turns TEXT,
                   last REAL,
                   active INTEGER DEFAULT 1)""")
        self.conn.commit()
        self.conn.commit()

    def persist_session(self, sid, title, opener, turns, last):
        """Save/refresh a live session so it survives gateway restarts."""
        import json as _j
        with self.lock:
            self.conn.execute(
                "INSERT INTO sessions(sid, title, opener, turns, last, active) "
                "VALUES(?,?,?,?,?,1) ON CONFLICT(sid) DO UPDATE SET "
                "title=excluded.title, turns=excluded.turns, last=excluded.last",
                (sid, title, opener, _j.dumps(turns), last))
            self.conn.commit()

    def close_persisted_session(self, sid):
        with self.lock:
            self.conn.execute(
                "UPDATE sessions SET active=0 WHERE sid=?", (sid,))
            self.conn.commit()

    def load_active_sessions(self):
        """Restore live sessions after a gateway restart."""
        import json as _j
        with self.lock:
            rows = self.conn.execute(
                "SELECT sid, title, opener, turns, last FROM sessions "
                "WHERE active=1 ORDER BY last DESC").fetchall()
        out = {}
        for sid, title, opener, turns_j, last in rows:
            try:
                turns = _j.loads(turns_j)
                out[sid] = {"turns": turns, "last": last,
                            "title": title, "opener": opener}
            except Exception:
                pass
        return out

    def add(self, task, action, G, title=None):
        with self.lock:
            self.conn.execute(
                "INSERT INTO cards(task, title, action, G, created) VALUES(?,?,?,?,?)",
                (task, title or task[:40], action, G, time.time()))
            self.conn.commit()

    def size(self):
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]

    def list_cards(self, offset=0, limit=50, q="", tag=""):
        """Paginated, searchable, tag-filtered list for the UI.

        tag: "" all / "positive" / "negative" / "pending" - same ±0.15
        thresholds the injector uses, so the UI shows exactly what
        injection believes."""
        where, params = [], []
        if q:
            where.append("(task LIKE ? OR action LIKE ?)")
            params += ["%" + q + "%"] * 2
        if tag == "positive":
            where.append("G > 0.15")
        elif tag == "negative":
            where.append("G < -0.15")
        elif tag == "pending":
            where.append("G BETWEEN -0.15 AND 0.15")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        with self.lock:
            total = self.conn.execute(
                "SELECT COUNT(*) FROM cards" + clause, params).fetchone()[0]
            rows = self.conn.execute(
                "SELECT id, task, title, action, G, created FROM cards" + clause +
                " ORDER BY id DESC LIMIT ? OFFSET ?",
                params + [limit, offset]).fetchall()
        return total, [dict(zip(("id", "task", "title", "action", "G", "created"), r))
                       for r in rows]

    def delete_card(self, card_id):
        with self.lock:
            cur = self.conn.execute("DELETE FROM cards WHERE id=?", (card_id,))
            self.conn.commit()
            return cur.rowcount > 0

    def stats_summary(self):
        """Dashboard aggregates: totals, today's delta, tag counts,
        score histogram (10 buckets), last-7-days daily counts."""
        day = 86400.0
        now = time.time()
        with self.lock:
            total, positive, negative = self.conn.execute(
                "SELECT COUNT(*),"
                " SUM(CASE WHEN G > 0.15 THEN 1 ELSE 0 END),"
                " SUM(CASE WHEN G < -0.15 THEN 1 ELSE 0 END) FROM cards"
            ).fetchone()
            today = self.conn.execute(
                "SELECT COUNT(*) FROM cards WHERE created > ?",
                (now - day,)).fetchone()[0]
            hist = [0] * 10
            for b, n in self.conn.execute(
                    "SELECT CAST((G + 1.0) * 5 AS INTEGER) AS b, COUNT(*)"
                    " FROM cards GROUP BY b").fetchall():
                b = min(max(int(b), 0), 9)
                hist[b] = n
            since = now - 7 * day
            daily = {time.strftime("%Y-%m-%d", time.localtime(since + i * day)): 0
                     for i in range(7)}
            for d, n in self.conn.execute(
                    "SELECT strftime('%Y-%m-%d', created, 'unixepoch',"
                    " 'localtime'), COUNT(*) FROM cards WHERE created > ?"
                    " GROUP BY 1", (since,)).fetchall():
                if d in daily:
                    daily[d] = n
        return {"total": total or 0, "today": today,
                "positive": positive or 0, "negative": negative or 0,
                "pending": (total or 0) - (positive or 0) - (negative or 0),
                "histogram": hist,
                "daily": [{"date": k, "count": v} for k, v in daily.items()]}

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


def _zcode_session_names():
    """Read ZCode's task index to map recent sessions to their names."""
    import sqlite3 as _sq
    import os as _os
    db = _os.path.expanduser("~/.zcode/v2/tasks-index.sqlite")
    if not _os.path.exists(db):
        return {}
    try:
        conn = _sq.connect("file:%s?mode=ro" % db, uri=True, timeout=1)
        rows = conn.execute(
            "SELECT task_id, title, workspace_path, updated_at FROM tasks "
            "WHERE deleted=0 AND archived=0 AND title != '' "
            "ORDER BY updated_at DESC LIMIT 200").fetchall()
        conn.close()
        return {r[0]: {"title": r[1], "workspace": r[2], "updated": r[3]}
                for r in rows}
    except Exception:
        return {}


class Gateway:
    """A transparent learning proxy. Work-channel routing:
    explicit --route > client config files > built-in public-cloud table.
    The judge channel is configured once at startup and scores every session
    regardless of which model is doing the work."""

    def __init__(self, judge_model, db_path,
                 judge_upstream, judge_api_key,
                 upstream=None, api_key=None,
                 routes=None, routes_from_clients=None, use_builtin_routes=True,
                 inject=True, extra_body=None, config=None):
        # fallback work channel (legacy single-upstream setups)
        self.upstream = (upstream or "").rstrip("/")
        self.api_key = api_key
        # judge channel: independent, global, one-time configuration
        self.judge_upstream = judge_upstream.rstrip("/")
        self.judge_api_key = judge_api_key
        self.judge_model = judge_model
        self.inject = inject
        self.extra_body = dict(extra_body or {})
        self.conf = dict(DEFAULTS)
        self.conf.update(config or {})
        self.mem = GatewayMemory(db_path)

        from earcon.gateway.routes import RouteTable
        self_port = None
        try:
            self_port = int(os.environ.get("EARCON_PORT", "0")) or None
        except ValueError:
            pass
        self.route_table = RouteTable(routes=routes or [],
                                      routes_from_clients=routes_from_clients or [],
                                      use_builtin=use_builtin_routes,
                                      self_port=self_port)

        self.sessions = {}            # sid -> {"turns": [...], "last": ts}
        self.closed_sessions = []     # ring buffer of recent closed summaries
        # restore live sessions from disk (survive gateway restarts)
        try:
            self.sessions = self.mem.load_active_sessions()
        except Exception:
            self.sessions = {}
        self.session_lock = threading.Lock()
        self._stop = threading.Event()
        self.reaper = threading.Thread(target=self._reap_loop, daemon=True)
        self.reaper.start()

    # ---------------- session bookkeeping ----------------

    def _after_sync(self, sid, opener):
        """After history sync: set opener/title and persist, mirroring
        record_turn's bookkeeping."""
        with self.session_lock:
            s = self.sessions.setdefault(sid, {"turns": [], "last": time.time()})
            if opener and "opener" not in s:
                s["opener"] = opener
        try:
            with self.session_lock:
                s = self.sessions[sid]
                self.mem.persist_session(
                    sid, s.get("title"), s.get("opener"),
                    [list(t) for t in s["turns"]], s["last"])
        except Exception:
            pass

    def record_turn(self, sid, user_text, assistant_text, opener=None):
        c = self.conf
        with self.session_lock:
            s = self.sessions.setdefault(sid, {"turns": [], "last": time.time()})
            if opener and "opener" not in s:
                s["opener"] = opener  # conversation opener: maps to ZCode name
            # persist for restart survival
            try:
                self.mem.persist_session(
                    sid, s.get("title"), s.get("opener"),
                    [list(t) for t in s["turns"]], s["last"])
            except Exception:
                pass
            # strip system-reminder BEFORE truncating: ZCode prepends long
            # skill lists to the real message, and truncation-first would
            # cut off the actual user content entirely
            clean_user = self._strip_system_raw(user_text)
            s["turns"].append((clean_user[:c["turn_user_chars"]],
                               (assistant_text or "")[:c["turn_assistant_chars"]]))
            s["last"] = time.time()
            # dynamic title: latest real user message wins
            # ZCode sends <system-reminder>...</system-reminder>real content
            # - extract the real content after the system tag
            import re as _re
            t = (user_text or "").strip()
            # strip system-reminder wrapper if present
            m = _re.match(r'^<system-reminder>.*?</system-reminder>\s*', t, _re.DOTALL)
            if m:
                t = t[m.end():].strip()
            # also skip pure system tags with no content
            if t and not t.startswith("<system"):
                s["title"] = t[:40] + ("..." if len(t) > 40 else "")

    def close_session(self, sid, reason="idle"):
        """Close a session: archive a summary for the UI, then judge async.

        Trajectories stay volatile (memory only, per the design), but the
        summary (turn count, when, how many cards were distilled) is kept
        so the UI can show 'this session was learned from'."""
        with self.session_lock:
            s = self.sessions.pop(sid, None)
        cards_before = self.mem.size()
        turns_data = (s or {}).get("turns", [])
        n_turns = len(turns_data)
        if s and n_turns >= self.conf["min_turns_to_judge"]:
            threading.Thread(
                target=self._judge_and_archive,
                args=(turns_data, sid, reason, n_turns, cards_before),
                daemon=True).start()
        elif s:
            self._archive_session(sid, turns_data, reason, cards=0)
        return n_turns

    def _judge_and_archive(self, turns, sid, reason, n_turns, cards_before):
        # extract title for cards (latest real user message)
        import re as _re
        title = sid
        for u, _ in reversed(turns):
            t = (u or "").strip()
            m = _re.match(r'^<system-reminder>.*?</system-reminder>\s*', t, _re.DOTALL)
            if m:
                t = t[m.end():].strip()
            if t and not t.startswith("<system"):
                title = t[:40]
                break
        self._judge_and_write(turns, title=title)
        self._archive_session(sid, turns, reason,
                              cards=self.mem.size() - cards_before)

    def _archive_session(self, sid, turns_data, reason, cards):
        """Archive a closed session with title + first turns for the UI.
        Title = latest real user message (topic drift: the end matters most)."""
        title = sid
        preview = []
        for i, (u, a) in enumerate(turns_data[:3]):
            preview.append({"user": self._strip_system(u)[:200],
                            "assistant": (a or "")[:200]})
        # scan from newest to oldest - latest real message wins
        import re as _re
        for u, _ in reversed(turns_data):
            t = (u or "").strip()
            # strip system-reminder wrapper, extract real content
            m = _re.match(r'^<system-reminder>.*?</system-reminder>\s*', t, _re.DOTALL)
            if m:
                t = t[m.end():].strip()
            if t and not t.startswith("<system"):
                title = t[:40] + ("..." if len(t) > 40 else "")
                break
        with self.session_lock:
            self.closed_sessions.append({
                "sid": sid, "title": title, "turns": len(turns_data),
                "closed_at": time.time(), "reason": reason, "cards": cards,
                "preview": preview})
            del self.closed_sessions[:-20]  # keep last 20

    @staticmethod
    def _strip_system_raw(text):
        """Strip system-reminder wrappers; return whatever real content
        remains (may be empty). Storage-time cleanup."""
        import re as _re
        t = (text or "").strip()
        t = _re.sub(r'<system-reminder>.*?</system-reminder>\s*', '', t,
                    flags=_re.DOTALL).strip()
        if t.startswith('<system-reminder>'):
            parts = t.split('\n\n')
            real = [p for p in parts
                    if p.strip() and not p.strip().startswith(('- ', '<', '#'))]
            t = real[-1].strip() if real else ""
        return t

    @staticmethod
    def _strip_system(text):
        """Remove <system-reminder>...</system-reminder> wrapper(s), keep real
        content. Handles unclosed tags (truncated skill lists) by falling
        back to the last blank-line-separated chunk that isn't system text."""
        import re as _re
        t = (text or "").strip()
        # strip all closed system-reminder blocks (there may be several)
        t = _re.sub(r'<system-reminder>.*?</system-reminder>\s*', '', t,
                    flags=_re.DOTALL).strip()
        # unclosed block: text starts with the tag but never closes it
        if t.startswith('<system-reminder>'):
            # try to find real content after the last skill entry: ZCode
            # skill dumps end with a line that has no leading '-'; the real
            # user message usually follows after a blank line
            parts = t.split('\n\n')
            real = [p for p in parts
                    if p.strip() and not p.strip().startswith(('- ', '<', '#'))]
            if real:
                t = real[-1].strip()
            else:
                t = t[-200:].strip() if len(t) > 200 else '(系统消息)'
        return t if t else "(系统消息)"

    def snapshot_sessions(self):
        """Lock-held view for the UI: live sessions + recent closed ones.
        Title is the first user message (truncated) - a real session summary
        instead of the opaque sid."""
        with self.session_lock:
            live = []
            for sid, s in self.sessions.items():
                preview = [{"user": self._strip_system(u)[:300],
                            "assistant": (a or "")[:300]}
                           for u, a in s["turns"]]
                # prefer the real ZCode session name, fall back to
                # dynamic title, then sid
                zname = _match_zcode_session(s.get("opener"))
                title = zname or s.get("title") or sid
                live.append({"sid": sid, "title": title,
                             "turns": len(s["turns"]), "last": s["last"],
                             "preview": preview})
            closed = list(self.closed_sessions)
        return {"live": live, "closed": list(reversed(closed))}

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

    def _judge_and_write(self, turns, title=None):
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
                                 int(m.group(2)) / 3.0, title=title)
        except Exception as e:
            print("[earcon] judge error:", e)

    def _sync_turns_from_history(self, sid, messages, assistant_reply):
        """Reconstruct session turns from the FULL message history clients
        like ZCode/Codex resend every request. Instead of recording only
        (first message, this turn's reply), replay all user/assistant pairs
        so the session shows every round. Latest turn pairs with the reply
        we just received."""
        import re as _re

        def _clean(t):
            t = (t or "").strip()
            t = _re.sub(r'<system-reminder>.*?</system-reminder>\s*', '', t,
                       flags=_re.DOTALL).strip()
            if t.startswith('<system-reminder>'):
                parts = t.split('\n\n')
                real = [p for p in parts
                        if p.strip() and not p.strip().startswith(('- ', '<', '#'))]
                t = real[-1].strip() if real else ""
            return t

        pairs = []  # (user, assistant) ordered
        pending_user = None
        for m in messages:
            role = m.get("role")
            c = m.get("content")
            text = ""
            if isinstance(c, str):
                text = c
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        break
            if role == "user":
                pending_user = text
            elif role == "assistant" and pending_user is not None:
                pairs.append((pending_user, text))
                pending_user = None
        # the final pending user pairs with the reply we just got
        if pending_user is not None:
            pairs.append((pending_user, assistant_reply or ""))

        if not pairs:
            return False

        c = self.conf
        with self.session_lock:
            s = self.sessions.setdefault(sid, {"turns": [], "last": time.time()})
            # history is authoritative: replace what we have (covers restarts
            # and missed turns); keep title/opener if already set
            new_turns = [(_clean(u)[:c["turn_user_chars"]],
                          (a or "")[:c["turn_assistant_chars"]])
                         for u, a in pairs]
            if len(new_turns) <= len(s["turns"]):
                # history got shorter (compaction?) - keep the longer local view
                return False
            s["turns"] = new_turns
            s["last"] = time.time()
            return True

    def _conv_opener(self, messages):
        """The first real user message of the conversation (full text),
        used to match the ZCode session name."""
        import re as _re
        for m in messages:
            if m.get("role") != "user":
                continue
            c = m.get("content")
            text = ""
            if isinstance(c, str):
                text = c
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        break
            text = text.strip()
            m2 = _re.match(r'^<system-reminder>.*?</system-reminder>\s*', text, _re.DOTALL)
            if m2:
                text = text[m2.end():].strip()
            if text and not text.startswith("<system"):
                return text
        return None

    def _task_signature(self, turns):
        import re as _re
        for u, _ in turns:
            t = (u or "").strip()
            if not t:
                continue
            # strip system-reminder wrapper, use real content
            m = _re.match(r'^<system-reminder>.*?</system-reminder>\s*', t, _re.DOTALL)
            if m:
                t = t[m.end():].strip()
            if t and not t.startswith("<system"):
                return t[:self.conf["task_sig_chars"]]
        return ""

    def _llm_chat(self, prompt, max_tokens=2000):
        payload = {"model": self.judge_model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "stream": False,
                   "temperature": 0.0}
        payload.update(self.extra_body)
        req = urllib.request.Request(
            self.judge_upstream + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + (self.judge_api_key or "")})
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

        # session id: explicit metadata wins; else derive from the conversation
        # history - clients like ZCode/Codex resend the full message list every
        # turn, so the first real user message is a stable conversation fingerprint.
        sid = (body.get("metadata") or {}).get("session_id")
        if not sid:
            fingerprint = _conversation_fingerprint(messages)
            sid = fingerprint or "default"

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

        # work channel: route by model name, with explicit > client-config >
        # built-in resolution. Upstream URL comes from the route table; the
        # key is the client's own unless the route provides one.
        model = body.get("model", "")
        try:
            target_upstream, route_key = self.route_table.resolve(model, auth_key)
        except KeyError:
            return {"error": {"no_route_for_model": model,
                              "hint": "add it: --route '%s=https://upstream/v1' "
                                      "or let it be read from client configs "
                                      "(--routes-from zcode,codex,hermes)" % model}}
        upstream_key = route_key or self.api_key or ""
        stream = body.get("stream", False)
        body_json = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json",
                   "Authorization": "Bearer " + upstream_key}
        req = urllib.request.Request(target_upstream + "/chat/completions",
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
                        opener = self._conv_opener(messages)
                        # rebuild ALL turns from the full history (clients
                        # resend everything each request), not just this one
                        if not self._sync_turns_from_history(sid, messages, "".join(collected)):
                            self.record_turn(sid, _first_user_text(messages) or "",
                                            "".join(collected), opener=opener)
                        else:
                            self._after_sync(sid, opener)

            return StreamingResponse(gen(), media_type="text/event-stream")
        else:
            d = json.loads(up_resp.read().decode("utf-8"))
            content = (d["choices"][0]["message"].get("content") or "")
            opener = self._conv_opener(messages)
            if not self._sync_turns_from_history(sid, messages, content):
                self.record_turn(sid, _first_user_text(messages) or "",
                                content, opener=opener)
            else:
                self._after_sync(sid, opener)
            return d


def _conversation_fingerprint(messages):
    """Derive a stable session id from the conversation history.
    The first REAL user message (skipping system reminders) is hashed -
    clients resend the full history each turn, so the first message stays
    constant within a conversation and differs across conversations."""
    import hashlib
    import re as _re
    for m in messages:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        text = ""
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    break
        text = text.strip()
        m2 = _re.match(r'^<system-reminder>.*?</system-reminder>\s*', text, _re.DOTALL)
        if m2:
            text = text[m2.end():].strip()
        if text and not text.startswith("<system"):
            # stable fingerprint from the conversation opener
            return "s-" + hashlib.sha1(text[:200].encode("utf-8")).hexdigest()[:10]
    return None


def _match_zcode_session(opener_text):
    """Match a gateway session to a ZCode session name by comparing the
    conversation opener to ZCode's task searchable_text prefix."""
    if not opener_text:
        return None
    prefix = opener_text.strip()[:200]
    import sqlite3 as _sq
    import os as _os
    import re as _re
    db = _os.path.expanduser("~/.zcode/v2/tasks-index.sqlite")
    try:
        conn = _sq.connect("file:%s?mode=ro" % db, uri=True, timeout=1)
        rows = conn.execute(
            "SELECT title, substr(searchable_text, 1, 250) FROM tasks "
            "WHERE deleted=0 AND archived=0 AND title != '' "
            "ORDER BY updated_at DESC LIMIT 200").fetchall()
        conn.close()
        for title, text_prefix in rows:
            t = (text_prefix or "").strip()
            m2 = _re.match(r'^<system-reminder>.*?</system-reminder>\s*', t, _re.DOTALL)
            if m2:
                t = t[m2.end():].strip()
            if t and t.startswith(prefix[:100]) or (prefix and prefix.startswith(t[:100])):
                return title
    except Exception:
        pass
    return None


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

    # CORS: the desktop client loads the UI from file:// (pywebview),
    # so API responses must be readable from a null origin. Local-only
    # single-user service; wide-open CORS is acceptable here.
    from starlette.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
        allow_headers=["*"])

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def proxy_chat(request: Request):
        body = await request.json()
        auth_key = request.headers.get("Authorization", "").replace("Bearer ", "")
        # handle_chat does blocking urllib calls (upstream timeout 300s) -
        # run it in starlette's threadpool to avoid freezing the event loop
        from starlette.concurrency import run_in_threadpool
        result = await run_in_threadpool(gateway.handle_chat, body, auth_key)
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
        n = gateway.close_session(sid, reason="manual")
        return JSONResponse({"closed": sid, "turns": n})

    # ---------------- UI API ----------------

    @app.get("/v1/earcon/ui/stats")
    async def ui_stats():
        return JSONResponse(gateway.mem.stats_summary())

    @app.get("/v1/earcon/ui/cards")
    async def ui_cards(offset: int = 0, limit: int = 50, q: str = "",
                       tag: str = ""):
        total, cards = gateway.mem.list_cards(offset=offset, limit=limit,
                                              q=q, tag=tag)
        return JSONResponse({"total": total, "cards": cards})

    @app.delete("/v1/earcon/ui/cards/{card_id}")
    async def ui_card_delete(card_id: int):
        ok = gateway.mem.delete_card(card_id)
        return JSONResponse({"deleted": ok})

    @app.get("/v1/earcon/ui/sessions")
    async def ui_sessions():
        return JSONResponse(gateway.snapshot_sessions())

    return app
