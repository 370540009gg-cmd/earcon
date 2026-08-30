# -*- coding: utf-8 -*-
import math

from earcon.core.advantage import AdvantageEstimator
from earcon.core.memory import MemoryStore, jaccard, tokenize


def test_jaccard_distinguishes_transposed_cells():
    # the classic pitfall: naive tokenization makes r0c1 and r1c0 identical
    a = tokenize("map:trap1|cell:r0c1|bag:empty")
    b = tokenize("map:trap1|cell:r1c0|bag:empty")
    assert jaccard(a, b) < 1.0


def test_retrieve_exact_state_prefers_own_cards(tmp_path):
    m = MemoryStore(str(tmp_path / "m.db"))
    m.add("ns", "cell:r0c0", "south", 0.9)
    m.add("ns", "cell:r1c1", "east", -0.5)
    got = m.retrieve("ns", "cell:r0c0")
    assert [a for a, _, _ in got] == ["south"]
    m.close()


def test_retrieve_recency_pool_ignores_stale_cards(tmp_path):
    m = MemoryStore(str(tmp_path / "m.db"), recency_pool=2)
    m.add("ns", "s", "old-good", 1.0)
    m.add("ns", "s", "new-bad", -1.0)
    m.add("ns", "s", "newest-good", 1.0)
    got = [a for a, _, _ in m.retrieve("ns", "s")]
    # the oldest card fell outside the recency pool
    assert "old-good" not in got
    m.close()


def test_namespace_isolation(tmp_path):
    m = MemoryStore(str(tmp_path / "m.db"))
    m.add("nsA", "s", "act", 1.0)
    assert m.retrieve("nsB", "s") == []
    m.close()


def test_estimator_shrinks_small_samples():
    est = AdvantageEstimator(seed=0)
    # 1 card for "a" (thin evidence) vs 10 cards for "b"
    nb = ([("a", 1.0, 1.0)] + [("b", -1.0, 1.0)] * 10)
    adv = est.estimate(nb, ["a", "b"])
    # without shrinkage the 1-card action would get a full +1 advantage
    assert adv["a"] < 1.0


def test_estimator_advantage_signs():
    est = AdvantageEstimator(seed=0)
    nb = [("good", 1.0, 1.0)] * 6 + [("bad", -1.0, 1.0)] * 6
    adv = est.estimate(nb, ["good", "bad", "unseen"])
    assert adv["good"] > 0 > adv["bad"]
    assert -1.0 <= adv["unseen"] <= 1.0


def test_estimator_clips():
    est = AdvantageEstimator(seed=0)
    nb = [("a", 100.0, 1.0)] * 5
    adv = est.estimate(nb, ["a", "b"])
    assert adv["a"] <= 1.0
