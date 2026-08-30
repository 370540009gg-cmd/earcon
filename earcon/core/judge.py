# -*- coding: utf-8 -*-
"""Judges: credit assignment at episode boundaries, then discounted returns.

- ReturnJudge: uses the environment's real per-step rewards (verifiable
  rewards - tests passing, tickets resolved). Default and recommended.
- LLMJudge: the paper's approach - a model scores each step of the
  trajectory. Quality of the judge bounds the whole system, so failures
  fall back to the verifiable path.
"""

import re

JUDGE_PROMPT = """You are scoring one episode of an agent's trajectory (outcome: {outcome}).
Rate each step's contribution to the final outcome on a scale of -3 to +3
(positive = helped, negative = hurt). Output exactly {n} integer scores,
one per line, nothing else.

{steps}"""


class ReturnJudge:
    """G_t = sum_u gamma^(u-t) * r_u, but normalized by the remaining
    horizon so that early steps in long episodes are not silently
    discounted into irrelevance. A plain discounted return makes G at
    t=0 of a 25-step failure ~-0.15 vs -1.0 at the end; the early
    trap-fall card then looks fine and memory never learns to avoid it.
    G_t = (raw discounted return) / (gamma^(T-t) scale), i.e. the
    average-quality-of-remaining-future interpretation."""

    def __init__(self, gamma=0.9):
        self.gamma = gamma

    def score(self, trajectory, success):
        """trajectory: [(state, action, r), ...] -> [(state, action, G), ...]"""
        T = len(trajectory)
        raw = []
        for t in range(T):
            G = sum((self.gamma ** (u - t)) * trajectory[u][2]
                    for u in range(t, T))
            raw.append(G / (1 - self.gamma) if self.gamma < 1.0 else G)
        out = [(s, a, g) for (s, a, _), g in zip(trajectory, raw)]
        return out


class LLMJudge:
    def __init__(self, backend, gamma=0.9, scale=3.0):
        self.backend = backend
        self.fallback = ReturnJudge(gamma)
        self.scale = scale

    def score(self, trajectory, success):
        steps = "\n".join(
            "%d. [%s] -> %s" % (i + 1, s, a)
            for i, (s, a, _) in enumerate(trajectory))
        prompt = JUDGE_PROMPT.format(
            outcome="success" if success else "failure",
            n=len(trajectory), steps=steps)
        try:
            text = self.backend.chat([{"role": "user", "content": prompt}])
            nums = [int(x) for x in re.findall(r"-?\d+", text)]
            if len(nums) < len(trajectory):
                raise ValueError("not enough scores")
            shaped = [(s, a, n / float(self.scale))
                      for (s, a, _), n in zip(trajectory, nums)]
            return self.fallback.score(shaped, success)
        except Exception:
            return self.fallback.score(trajectory, success)
