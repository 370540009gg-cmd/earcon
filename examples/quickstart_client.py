# -*- coding: utf-8 -*-
"""Quickstart: point any OpenAI SDK at the earcon gateway.

Terminal 1 (only the judge channel is configured here):
    earcon serve --judge-upstream https://api.deepseek.com/v1 \
        --judge-api-key $KEY --judge-model deepseek-chat

Terminal 2:
    python examples/quickstart_client.py

The only change vs a vanilla OpenAI app is base_url; your real upstream
URL and API key stay wherever this client already has them. Run the
script a few times with the same task phrasing, close the session
(waits for the timeout, or POST /v1/earcon/session/default/close), and
watch the injected experience change the responses.
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
