# -*- coding: utf-8 -*-
"""Policy: z' = z + beta * A, the closed-form update of KL-constrained
policy improvement. Works over any backend that produces per-option
"impulse values" (logits or pseudo-logits).

use_memory=False degrades to the pure base policy (the control arm).
"""

import math
import random


class JitRLPolicy:
    def __init__(self, backend, memory, estimator, beta=3.0,
                 use_memory=True, greedy=False, seed=0):
        self.backend = backend
        self.memory = memory
        self.estimator = estimator
        self.beta = beta
        self.use_memory = use_memory
        self.greedy = greedy
        self.rng = random.Random(seed)

    def act(self, prompt, state, namespace, actions):
        """Returns (chosen action, debug info). actions: list of texts."""
        options = [(chr(ord("A") + i), a) for i, a in enumerate(actions)]
        z = self.backend.choice_logits(prompt, options)

        adv = {a: 0.0 for a in actions}
        n_neighbors = 0
        if self.use_memory:
            neighbors = self.memory.retrieve(namespace, state)
            n_neighbors = len(neighbors)
            adv = self.estimator.estimate(neighbors, actions)

        adjusted = {
            letter: z[letter] + self.beta * adv[text]
            for letter, text in options
        }

        best = (max(adjusted, key=adjusted.get) if self.greedy
                else self._softmax_sample(adjusted))

        debug = {
            "neighbors": n_neighbors,
            "logits": {t: round(z[l], 2) for l, t in options},
            "advantage": {t: round(adv[t], 2) for _, t in options},
            "chosen": options[ord(best) - ord("A")][1],
        }
        return debug["chosen"], debug

    def _softmax_sample(self, logits):
        m = max(logits.values())
        exps = {k: math.exp(v - m) for k, v in logits.items()}
        total = sum(exps.values())
        r = self.rng.random() * total
        acc = 0.0
        for k, v in exps.items():
            acc += v
            if r <= acc:
                return k
        return list(logits)[-1]
