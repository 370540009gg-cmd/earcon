# -*- coding: utf-8 -*-
"""JitRL loop on the treasure maze (the paper's discrete-action path).

Mock backend needs nothing installed:
    python examples/treasure_maze/run.py --episodes 150 --seed 2 --beta 6 --k 15

Control arm (no memory) vs learning arm:
    python examples/treasure_maze/run.py --episodes 150 --no-memory --db baseline.db

Real backend (any OpenAI-compatible endpoint with logprobs, e.g. vLLM):
    python examples/treasure_maze/run.py --backend openai \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen3-8B
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from earcon.core.advantage import AdvantageEstimator
from earcon.core.backends import MockBackend, OpenAICompatBackend, VerbalConfidenceBackend
from earcon.core.judge import LLMJudge, ReturnJudge
from earcon.core.memory import MemoryStore
from earcon.core.policy import JitRLPolicy
from examples.treasure_maze.env import ALL_ACTIONS, MAPS, TreasureMaze, optimal_steps

PROMPT_TMPL = """You are playing a text treasure game. Goal: find the key,
then use it to open the chest.
{obs}
{history}
Choose the next action. Repeating a route that just failed is pointless."""


def run_episode(env, policy, verbose=False, history_len=12):
    obs = env.reset()
    trajectory, history = [], []
    while not env.done:
        state = env.abstract_state()
        h = ("Recent steps:\n" + "\n".join(
            "%d. %s -> %s" % (i + 1, a, e) for i, (a, e) in enumerate(history))
             if history else "(first step of this episode)")
        action, debug = policy.act(
            PROMPT_TMPL.format(obs=obs, history=h), state, env.map_id, ALL_ACTIONS)
        obs, _, _, reward = env.step(action)
        trajectory.append((state, action, reward))
        if env.last_event:
            history.append((action, env.last_event))
            history = history[-history_len:]
        if verbose:
            print("    [%s] neighbors:%d chose:%s r:%+.1f adv:%s"
                  % (state, debug["neighbors"], action, reward, debug["advantage"]))
    return trajectory, env.success, env.steps, env.trap_falls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["mock", "openai", "verbal"],
                   default="mock")
    p.add_argument("--base-url", default=os.environ.get("EARCON_UPSTREAM", ""))
    p.add_argument("--api-key", default=os.environ.get("EARCON_API_KEY", ""))
    p.add_argument("--model", default="")
    p.add_argument("--judge", choices=["return", "llm"], default="return")
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--maps", default="trap1,trap2")
    p.add_argument("--beta", type=float, default=6.0)
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--db", default="earcon_memory.db")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.backend == "mock":
        backend = MockBackend(seed=args.seed)
    elif args.backend == "verbal":
        backend = VerbalConfidenceBackend(args.base_url, args.api_key, args.model)
    else:
        backend = OpenAICompatBackend(args.base_url, args.api_key, args.model)

    memory = MemoryStore(args.db)
    estimator = AdvantageEstimator(lam=args.lam, alpha=args.alpha, seed=args.seed)
    policy = JitRLPolicy(backend, memory, estimator, beta=args.beta,
                         use_memory=not args.no_memory,
                         greedy=args.greedy, seed=args.seed)
    judge = LLMJudge(backend) if args.judge == "llm" else ReturnJudge()

    map_ids = args.maps.split(",")
    results = []  # (success, steps, trap_falls)
    print("earcon maze | backend=%s memory=%s beta=%.1f | optimal=%d steps"
          % (args.backend, "OFF (control)" if args.no_memory else "ON",
             args.beta, optimal_steps(map_ids[0])))
    print("-" * 64)

    for ep in range(1, args.episodes + 1):
        env = TreasureMaze(map_ids[(ep - 1) % len(map_ids)],
                           max_steps=args.max_steps)
        traj, success, steps, falls = run_episode(env, policy, args.verbose)
        results.append((success, steps, falls))

        if not args.no_memory:
            for state, action, G in judge.score(traj, success):
                ns = state.split("|")[0].split(":")[1]
                memory.add(ns, state, action, G)

        print("ep%3d [%s] %s %2d steps, %d trap falls | memory %d cards"
              % (ep, env.map_id, "WIN " if success else "FAIL", steps,
                 falls, memory.size()))
        if ep % 10 == 0:
            w = results[-10:]
            print("  >> last 10: win %3.0f%% | steps %.1f | falls %.1f"
                  % (100 * sum(1 for s, _, _ in w if s) / len(w),
                     sum(t for _, t, _ in w) / len(w),
                     sum(f for _, _, f in w) / len(w)))

    print("-" * 64)
    total = sum(1 for s, _, _ in results if s)
    win_steps = [t for s, t, _ in results if s]
    print("total: win %.1f%% | winning avg steps %.1f (optimal %d) | "
          "avg falls %.1f"
          % (100 * total / len(results),
             sum(win_steps) / len(win_steps) if win_steps else 0,
             optimal_steps(map_ids[0]),
             sum(f for _, _, f in results) / len(results)))
    memory.close()


if __name__ == "__main__":
    main()
