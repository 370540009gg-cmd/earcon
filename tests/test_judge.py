# -*- coding: utf-8 -*-
"""Judge: horizon-normalized returns, LLM judge fallback."""

from earcon.core.judge import LLMJudge, ReturnJudge


class FlakyBackend:
    def __init__(self, text=None, fail=False):
        self.text = text
        self.fail = fail

    def chat(self, messages, max_tokens=1024):
        if self.fail:
            raise RuntimeError("backend down")
        return self.text


def test_horizon_normalization_keeps_early_steps_signal():
    traj = [("s", "a", 0.0)] * 20 + [("s", "a", -0.3)] * 2 + [("s", "a", -1.0)]
    scored = ReturnJudge().score(traj, success=False)
    # early G must not be diluted to near-zero (the bug that stalled learning)
    assert abs(scored[0][2]) > 0.4


def test_llm_judge_parses_scores():
    traj = [("s1", "a1", 0.0), ("s2", "a2", 0.0), ("s3", "a3", 0.0)]
    backend = FlakyBackend(text="2\n-1\n3")
    # gamma=0: each card's G is exactly its judged score (no tail rewards)
    scored = LLMJudge(backend, gamma=0.0, scale=3.0).score(traj, success=True)
    assert scored[0][2] > 0 and scored[1][2] < 0 and scored[2][2] > scored[0][2]


def test_llm_judge_falls_back_when_parse_fails():
    traj = [("s1", "a1", 0.0), ("s2", "a2", 0.0), ("s3", "a3", 0.0)]
    backend = FlakyBackend(text="I think the agent did fine overall")
    scored = LLMJudge(backend).score(traj, success=True)
    # fallback: verifiable-return path still produces cards
    assert len(scored) == 3
    assert all(isinstance(G, float) for _, _, G in scored)


def test_llm_judge_falls_back_when_backend_raises():
    traj = [("s1", "a1", 0.5), ("s2", "a2", 0.5)]
    scored = LLMJudge(FlakyBackend(fail=True)).score(traj, success=True)
    assert len(scored) == 2
