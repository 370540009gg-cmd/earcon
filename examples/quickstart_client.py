# -*- coding: utf-8 -*-
"""Quickstart: point any OpenAI SDK at the earcon gateway.

Terminal 1:
    earcon serve --upstream https://api.openai.com/v1 \
        --api-key $OPENAI_API_KEY --judge-model gpt-4o-mini

Terminal 2:
    python examples/quickstart_client.py

The only change vs a vanilla OpenAI app is base_url. Run this script a
few times with the same task phrasing, close the session (waits for the
timeout, or POST /v1/earcon/session/default/close), and watch the
injected experience change the responses.
"""

import os

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8800/v1",
    api_key=os.environ.get("EARCON_API_KEY", "unused-by-earcon"),
)

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in your own words."}],
)
print(resp.choices[0].message.content)
