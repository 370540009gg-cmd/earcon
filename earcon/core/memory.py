# -*- coding: utf-8 -*-
"""Experience memory: (state, action, return) cards in SQLite, Jaccard retrieval.

Non-parametric memory in the JitRL sense. Structured-text + Jaccard was
empirically better than embeddings in our experiments (and needs no
vector infra), so retrieval is deliberately simple. Two corrections the
paper leaves implicit are built in:

- recency pool: experience is non-stationary; recent cards must win ties
  against early failures (otherwise learning curves can stall and even
  regress as stale cards keep occupying the neighborhood);
- single-token coordinates: naive tokenization can treat "r0c1" and
  "r1c0" as the same token set, silently conflating distinct states.
  Callers should compose position-like attributes into one token.
"""

import re
import sqlite3
import time

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]")


def tokenize(text):
    """Latin/digits by word, CJK by character; returns a set for Jaccard."""
    return set(_TOKEN_RE.findall(text or ""))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / float(len(a) + len(b) - inter)


class MemoryStore:
    def __init__(self, db_path="earcon_memory.db", max_cards=5000,
                 recency_pool=2000):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = __import__("threading").Lock()
        self.max_cards = max_cards
        self.recency_pool = recency_pool
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS cards(
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   namespace TEXT,
                   state TEXT,
                   action TEXT,
                   G REAL,
                   created REAL)""")
        self.conn.commit()

    def add(self, namespace, state, action, G):
        with self.lock:
            self.conn.execute(
                "INSERT INTO cards(namespace, state, action, G, created) "
                "VALUES(?,?,?,?,?)",
                (namespace, state, action, G, time.time()))
            self.conn.commit()
            n = self.conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
            if n > self.max_cards:
                self.conn.execute(
                    "DELETE FROM cards WHERE id IN "
                    "(SELECT id FROM cards ORDER BY id LIMIT ?)",
                    (n - self.max_cards,))
                self.conn.commit()

    def retrieve(self, namespace, state, k=10, min_sim=0.9):
        """Most-similar k cards as [(action, G, sim), ...].

        min_sim defaults high (near-exact match) for discrete-state tasks;
        lower it for fuzzier domains. Ties prefer newer cards.
        """
        rows = self.conn.execute(
            "SELECT state, action, G FROM cards WHERE namespace=? "
            "ORDER BY id DESC LIMIT ?",
            (namespace, self.recency_pool)).fetchall()
        if not rows:
            return []
        q = tokenize(state)
        scored = [(action, G, jaccard(q, tokenize(s))) for s, action, G in rows]
        scored = [x for x in scored if x[2] >= min_sim]
        scored.sort(key=lambda x: -x[2])
        return scored[:k]

    def size(self, namespace=None):
        if namespace is None:
            return self.conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        return self.conn.execute(
            "SELECT COUNT(*) FROM cards WHERE namespace=?",
            (namespace,)).fetchone()[0]

    def close(self):
        self.conn.close()
