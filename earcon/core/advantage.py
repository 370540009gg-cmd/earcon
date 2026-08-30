# -*- coding: utf-8 -*-
"""Advantage estimation: V(s), Q(s,a), A = Q - V, UCB exploration, shrinkage.

Corresponds to step 3 of JitRL:
  V   = mean return over the neighborhood        ("how good is this situation")
  Q   = mean return of the cards that took action a
  A   = Q - V                                    ("how much better than average")
  unseen actions: UCB-style optimism with probability lam, else 0

Two engineering corrections the paper leaves implicit:

- small-sample shrinkage: A = (Q - V) * n / (n + m). An action with 1-2
  cards has a noisy Q; max-normalization amplified that noise into a
  full-scale advantage (we observed "bump into a wall" being learned as
  the best action). Shrinkage lets thin evidence speak softly.
- clip to [-1, 1] instead of max-normalizing: normalization re-amplifies
  relative noise when all advantages are small, cancelling shrinkage.
"""

import random


class AdvantageEstimator:
    def __init__(self, lam=0.05, alpha=2.0, min_neighbors=3, shrink=4.0, seed=0):
        self.lam = lam          # exploration probability
        self.alpha = alpha      # UCB bonus strength
        self.min_neighbors = min_neighbors
        self.shrink = shrink
        self.rng = random.Random(seed)

    def estimate(self, neighbors, actions):
        """neighbors: [(action, G, sim), ...]; actions: full action list.
        Returns {action: A} clipped to [-1, 1]."""
        n = len(neighbors)
        if n < self.min_neighbors:
            return {a: 0.0 for a in actions}

        V = sum(G for _, G, _ in neighbors) / n
        adv = {}
        for a in actions:
            Gs = [G for act, G, _ in neighbors if act == a]
            if Gs:
                Q = sum(Gs) / len(Gs)
                adv[a] = (Q - V) * len(Gs) / (len(Gs) + self.shrink)
            else:
                if self.rng.random() < self.lam:
                    adv[a] = self.alpha / n
                else:
                    adv[a] = 0.0

        return {a: max(-1.0, min(1.0, v)) for a, v in adv.items()}
