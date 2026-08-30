# -*- coding: utf-8 -*-
"""Model backends: produce per-option "impulse values" (logit or pseudo-logit).

- MockBackend: zero-dependency stand-in with human-like biases, for demos
  and tests. It has direction preferences, reacts to objects in sight
  (including distractions), and has no cross-episode memory - exactly the
  flaws a frozen base model exhibits in a new environment.
- OpenAICompatBackend: real logits via logprobs (the paper's token-level
  path; requires an OpenAI-compatible endpoint that supports logprobs,
  e.g. a self-hosted vLLM). Needs the `openai` package.
- VerbalConfidenceBackend: black-box fallback for restricted gateways
  (no logprobs): one call asking the model to rate each option 0-100,
  ln(p) as a pseudo-logit. The paper's black-box path. Any vendor-
  private parameters (e.g. disabling a reasoning mode) go through
  `extra_body` and are never sent by default.
"""

import json
import math
import re


class MockBackend:
    PREFS = {
        "north": 1.2, "east": 1.0, "south": 0.2, "west": 0.0,
        "pick key": -0.5, "take treasure": -0.5, "smash jar": -0.5,
    }
    REFLEX = {
        "a key on the ground": {"pick key": 2.2},
        "you have the key": {"take treasure": 2.2},
        "a chest stands here": {"take treasure": 0.3},
        "a jar sits in the corner": {"smash jar": 0.8},
    }

    def __init__(self, seed=0, noise=1.0):
        import random
        self.rng = random.Random(seed)
        self.noise = noise

    def choice_logits(self, prompt, options):
        """options: [(letter, text), ...] -> {letter: logit}"""
        reflex = {}
        low = prompt.lower()
        for marker, boosts in self.REFLEX.items():
            if marker in low:
                for k, v in boosts.items():
                    reflex[k] = reflex.get(k, 0.0) + v
        return {
            letter: (self.PREFS.get(text, 0.0) + reflex.get(text, 0.0)
                     + self.rng.gauss(0, self.noise))
            for letter, text in options
        }


class OpenAICompatBackend:
    def __init__(self, base_url, api_key, model, timeout=60, retries=2):
        from openai import OpenAI  # lazy: core lib stays zero-dependency
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY",
                             timeout=timeout, max_retries=retries)
        self.model = model

    def choice_logits(self, prompt, options):
        lines = [prompt, "", "Options:"]
        for letter, text in options:
            lines.append("%s. %s" % (letter, text))
        lines.append("Answer with a single letter, nothing else.")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "\n".join(lines)}],
            max_tokens=1, temperature=0.0,
            logprobs=True, top_logprobs=20)
        top = resp.choices[0].logprobs.content[0].top_logprobs
        logits = {}
        for letter, _ in options:
            hit = [t.logprob for t in top if t.token.strip().upper() == letter]
            if hit:
                logits[letter] = max(hit)
        if not logits:
            return {letter: 0.0 for letter, _ in options}
        floor = min(logits.values()) - 5.0
        return {letter: logits.get(letter, floor) for letter, _ in options}

    def chat(self, messages, max_tokens=1024):
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens)
        return resp.choices[0].message.content


class VerbalConfidenceBackend:
    """Black-box path: one scoring call, JSON output, ln(p) pseudo-logits."""

    def __init__(self, base_url, api_key, model, timeout=60, retries=2,
                 extra_body=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.extra_body = dict(extra_body or {})

    def _post(self, payload):
        import urllib.request
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + (self.api_key or "")})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _chat(self, text, max_tokens):
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": text}],
                   "max_tokens": max_tokens, "stream": False}
        payload.update(self.extra_body)
        d = self._post(payload)
        return d["choices"][0]["message"].get("content") or ""

    def choice_logits(self, prompt, options):
        lines = [prompt, "", "Options:"]
        for letter, text in options:
            lines.append("%s. %s" % (letter, text))
        lines.append("Rate each option 0-100 (higher = more you want to do it)."
                     " Output one line of JSON only, e.g. %s"
                     % json.dumps({l: 50 for l, _ in options}))
        scores = None
        for _ in range(self.retries + 1):
            try:
                text = self._chat("\n".join(lines), max_tokens=3000)
                m = re.search(r"\{[^{}]*\}", text, re.S)
                scores = json.loads(m.group(0))
                break
            except Exception:
                continue
        if not isinstance(scores, dict):
            return {letter: 0.0 for letter, _ in options}
        logits = {}
        for letter, _ in options:
            try:
                s = float(scores.get(letter, 1))
            except (TypeError, ValueError):
                s = 1.0
            logits[letter] = math.log(max(s, 0.5) / 100.0)
        return logits

    def chat(self, messages, max_tokens=4096):
        text = "\n\n".join(m["content"] for m in messages)
        return self._chat(text, max_tokens=max_tokens)
