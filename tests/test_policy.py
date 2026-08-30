# -*- coding: utf-8 -*-
"""Policy math: z' = z + beta*A, softmax sampling, memory on/off."""

from earcon.core.advantage import AdvantageEstimator
from earcon.core.backends import MockBackend
from earcon.core.memory import MemoryStore
from earcon.core.policy import JitRLPolicy

ACTIONS = ["north", "east", "south", "west"]


class FixedBackend:
    def __init__(self, logits):
        self.logits = logits

    def choice_logits(self, prompt, options):
        return {letter: self.logits.get(text, 0.0) for letter, text in options}


def _policy(tmp_path, backend, **kw):
    mem = MemoryStore(str(tmp_path / "p.db"))
    est = AdvantageEstimator(seed=0)
    return JitRLPolicy(backend, mem, est, **kw), mem


def test_logit_update_pulls_preferred_action(tmp_path):
    # base prefers east (z=2) over south (z=0); memory says south is better
    backend = FixedBackend({"east": 2.0, "south": 0.0})
    pol, mem = _policy(tmp_path, backend, beta=3.0, greedy=True, seed=0)
    for _ in range(10):
        mem.add("ns", "s", "south", 1.0)
        mem.add("ns", "s", "east", -1.0)
    action, debug = pol.act("prompt", "s", "ns", ACTIONS)
    assert action == "south"  # beta*A flipped the ranking
    assert debug["logits"]["east"] > debug["logits"]["south"]


def test_no_memory_is_pure_base(tmp_path):
    backend = FixedBackend({"east": 2.0, "south": 0.0})
    pol, mem = _policy(tmp_path, backend, use_memory=False, greedy=True)
    for _ in range(10):
        mem.add("ns", "s", "south", 1.0)
    action, _ = pol.act("prompt", "s", "ns", ACTIONS)
    assert action == "east"


def test_beta_zero_ignores_memory(tmp_path):
    backend = FixedBackend({"east": 2.0, "south": 0.0})
    pol, mem = _policy(tmp_path, backend, beta=0.0, greedy=True)
    for _ in range(10):
        mem.add("ns", "s", "south", 1.0)
        mem.add("ns", "s", "east", -1.0)
    action, _ = pol.act("prompt", "s", "ns", ACTIONS)
    assert action == "east"


def test_softmax_samples_all_actions(tmp_path):
    backend = FixedBackend({a: 0.0 for a in ACTIONS})
    pol, _ = _policy(tmp_path, backend, beta=1.0, greedy=False, seed=1)
    seen = set()
    for _ in range(200):
        a, _ = pol.act("prompt", "s", "ns", ACTIONS)
        seen.add(a)
    assert seen == set(ACTIONS)
