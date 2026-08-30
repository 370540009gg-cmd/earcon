# Release checklist

Do these before/while publishing. Generated during repo preparation -
tick as you go.

## Security (do first)

- [ ] **Revoke the internal API key** that appeared in the pre-open-source
      docs (`5666589dc…`). It was written to disk; treat it as leaked.
- [ ] Confirm this repo has no leftover `.db` / `.log` / `.env` files
      (`git ls-files | grep -E '\.(db|log|env)$'` must be empty).
- [ ] Re-scan after any future edits:
      `grep -rn -iE "llm-gw|jd\.local|joybuilder|<key>" .`

## Naming

- [ ] `earcon` is a working title. Before publishing, check GitHub name
      collisions (`https://github.com/earcon`) and PyPI
      (`pip index versions earcon`). If taken, rename globally:
      repo dir, `pyproject.toml [project] name`, package dir, CLI name
      in `[project.scripts]`, and the `earcon experience` injection
      marker string (keep them consistent).
- [ ] Decide author name/email in `pyproject.toml` (currently absent
      on purpose) and for the GitHub repo.

## Publishing

- [ ] Create the GitHub repo (public), push:
      `git remote add origin git@github.com:<you>/earcon.git && git push -u origin main`
- [ ] Add topics on the repo page: `llm`, `agents`, `memory`,
      `self-improvement`, `openai-compatible`, `jitrl`.
- [ ] Optional: publish to PyPI (`pip install build && python -m build
      && twine upload dist/*`) - or wait until someone asks.
- [ ] Optional CI: GitHub Actions running `pytest` on 3.9-3.12.

## Launch (pick 1-2, don't spam all at once)

- HN: "Show HN: Earcon – a self-evolving memory proxy for LLM apps
  (change one base_url)". Lead with the 2%→43% maze result and the
  honest limitations section - that's the credibility hook.
- r/LocalLLaMA: emphasize the vLLM/logprobs path and "works with any
  OpenAI-compatible endpoint".
- The JitRL paper's community (Twitter/X thread citing
  arXiv:2601.18510, ICML) - the paper's authors are the most likely
  early adopters.

## Post-release roadmap (see README)

- Hermes agent `MemoryProvider` adapter (their CONTRIBUTING explicitly
  routes memory providers to standalone repos + pip entry points - the
  `hermes_agent.memory_providers` group).
- vLLM `logit_bias` server-side steering.
- Card curation CLI (`earcon cards list/edit/delete`).
- Multi-namespace memory.

## Things deliberately not in v0.1

- No CI, no PyPI publish, no Hermes adapter, no docker image - ship
  small, iterate with real users.
