# Client integration

earcon speaks the OpenAI Chat Completions protocol both ways, so
anything that can set a base URL can use it. The one decision: how
sessions are identified.

## Session identity

- **Default**: every request lands in session `default`; it closes
  after `--session-timeout` seconds of inactivity (default 1800) or on
  explicit `POST /v1/earcon/session/<sid>/close`.
- **Explicit**: pass a `metadata.session_id` field in the request body
  to group turns into named sessions that map to your app's notion of
  a conversation/ticket.

## OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8800/v1",
                api_key="unused")  # gateway rewrites auth
resp = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "fix the flaky test"}],
    extra_body={"metadata": {"session_id": "ticket-123"}})
```

## curl

```bash
curl http://127.0.0.1:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"your-model","messages":[{"role":"user","content":"hi"}]}'

# close a session on demand (judging happens right after)
curl -X POST http://127.0.0.1:8800/v1/earcon/session/ticket-123/close
```

## LangGraph / LangChain

Configure the ChatOpenAI-compatible node with
`base_url="http://127.0.0.1:8800/v1"` and any key. Nothing else
changes.

## Coding-agent clients (Claude Code, Codex, etc.)

Most such clients let you point an "OpenAI-compatible provider" at a
custom base URL. Set it to `http://127.0.0.1:8800/v1`, pick any model
name your upstream serves, and the sessions flow through earcon. If
the client sends its own per-provider key, earcon ignores it and uses
its own `--api-key` for upstream calls.

## Streaming

Fully transparent: chunks are forwarded as-is while the assistant text
is accumulated in the background for the session buffer. Both SSE
(`stream: true`) and non-streaming requests work.

## Operation

| endpoint | purpose |
|---|---|
| `POST /v1/chat/completions` | the proxy itself (also at `/chat/completions`) |
| `GET /v1/models` | passthrough of the upstream model list |
| `GET /v1/earcon/stats` | card count, open sessions, inject mode |
| `POST /v1/earcon/session/<sid>/close` | close + judge a session now |

Inspect and curate memory with plain SQLite:

```bash
sqlite3 earcon_memory.db "SELECT id, task, action, G, created FROM cards ORDER BY id DESC LIMIT 20"
# remove a bad card
sqlite3 earcon_memory.db "DELETE FROM cards WHERE id = 42"
```

## A note on privacy

Everything the judge sees (and stores) passes through your upstream
LLM already - earcon adds one more LLM call per session. The card
store is a local file you fully control. If your sessions contain
secrets, treat `earcon_memory.db` like a credentials file.
