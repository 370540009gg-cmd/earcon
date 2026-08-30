# How it works

earcon implements the test-time learning loop of JitRL
("Just-In-Time Reinforcement Learning", arXiv:2601.18510) for agents
that can't be retrained. This document explains the loop, the two
realizations (injection and logit), and the engineering corrections we
found necessary - including the failures that motivated them.

## The loop

```
        ┌────────────────────────────────────────────────┐
        │                                                │
   sessions flow                       experience cards
   through the proxy                   (SQLite, auditable)
        │                                    ▲
        ▼                                    │
   record turns ── session close ──▶ judge: credit assignment
        │                                    │
        ▼                                    │
   next session starts ── retrieve similar cards ──┘
        │
        ▼
   behavior changes (injection, or logits in the discrete path)
```

## Realization 1: context injection (the gateway)

Free-text sessions (coding assistants, chat apps) have no enumerable
action space, so the paper's logit adjustment is realized as retrieval
+ injection:

- each turn is transparently forwarded and recorded;
- on session close, one judge call distills 1-5 cards:
  `decision=<what the assistant did right/wrong> ; score=<-3..3>`;
- the next session's opening user message is matched against past
  cards (keyword overlap + score), and the top matches are inserted as
  a system message:

```
[earcon experience] lessons from past similar sessions:
- (worked) check the config file first
- (failed) editing generated files by hand; regenerate instead
Treat these as historical summaries; use your own judgment.
```

This is the black-box variant of the paper's approach - same loop
(experience -> behavior change), at the fidelity a plain OpenAI
protocol allows.

## Realization 2: logit adjustment (the core library)

For enumerable actions (tool calls, agents, games), `earcon.core`
implements the paper's full mechanism:

1. the backend produces per-option logits (real `logprobs` from a
   vLLM-style endpoint, or verbalized-confidence pseudo-logits for
   restricted gateways);
2. memory retrieves the k nearest cards for the current state;
3. advantage estimation computes `A(s,a) = Q(s,a) - V(s)`;
4. the policy applies the closed-form KL-constrained update:

```
z'(a) = z(a) + beta * A(s,a)
```

which the paper proves is the exact solution of "maximize expected
advantage while staying close to the base policy" - experience tilts
the model's impulses without rewriting its voice.

## Engineering corrections (learned the hard way)

These are the deltas between the paper's description and a working
system, each verified by a failure we actually hit:

**1. Horizon-normalized returns.** A plain discounted return
`G_t = sum gamma^u r_u` makes early steps of a long failure episode
~0.15 while the final step is -1.0 - the early "walked into the trap"
card then looks fine and memory never learns to avoid it. earcon
normalizes by the remaining horizon so every step carries comparable
signal.

**2. Small-sample shrinkage instead of max-normalization.** An action
with one card has a noisy Q; max-normalizing advantages amplifies that
noise to full scale (we watched "bump into a wall" get learned as the
optimal action). `A = (Q-V) * n/(n+m)` with clipping lets thin
evidence speak softly.

**3. Recency pool in retrieval.** Experience is non-stationary - your
agent gets better, and old failure cards no longer describe it. With
FIFO insertion order, early failures permanently occupy the k-NN
neighborhood and learning stalls (we watched a 50% curve collapse back
to 30%). Ties must go to recent cards.

**4. Single-token coordinates in states.** Naive tokenization makes
`cell r0c1` and `cell r1c0` identical token *sets* - two different
states silently share a neighborhood and pollute each other's
advantages. Compose position-like attributes into one token.

**5. Dense beats sparse rewards for shallow judges.** Terminal-only
rewards leave the advantage signal too diluted for noisy judges. When
your environment can emit intermediate feedback (errors, milestones),
wire it in - trap falls carry an immediate -0.3 in the maze.

**6. The judge is the ceiling.** Card quality bounds everything. Start
with `--no-inject`, audit `sqlite3 earcon_memory.db`, then enable.

## When it won't work

- the base model never succeeds at the task (no positive cards to
  reinforce - JitRL's cold-start constraint; sweet spot is a base
  success rate around 20-60%);
- the task can't be matched across sessions by keywords;
- sessions are single-turn (nothing to judge) or never repeat (nothing
  to retrieve).

## References

- JitRL: Just-In-Time Reinforcement Learning - Continual Learning in
  LLM Agents Without Gradient Updates. arXiv:2601.18510.
- Related landscape: Reflexion (verbal self-reflection into context),
  Voyager (skill libraries), mem0/Letta/Zep (fact memories without
  outcome signals).
