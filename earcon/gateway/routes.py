# -*- coding: utf-8 -*-
"""Model → upstream routing, with three-tier resolution.

Priority (highest wins):
  1. explicit --route flags given at startup (user's final say)
  2. routes read from client config files (zcode/codex/hermes)
  3. built-in table for well-known public providers

The built-in table is a starting point only - it covers public clouds by
key-prefix and well-known model naming. Anything internal (custom gateways,
private deployments, -vendor-suffixed model names) cannot be guessed and
must come from client configs or explicit routes.
"""

import json
import os

# Built-in routing by API-key prefix (covers most public clouds).
# {prefix: upstream base url}
BUILTIN_KEY_PREFIXES = {
    "sk-or-": "https://openrouter.ai/api/v1",
    "sk-ant-": "https://api.anthropic.com",   # note: anthropic protocol, not OpenAI
    "sk-": "https://api.openai.com/v1",
    "ak-": "https://api.deepseek.com/v1",
}
# Well-known model names that imply a provider regardless of key prefix.
# {model substring: upstream base url}
BUILTIN_MODEL_HINTS = {
    "deepseek": "https://api.deepseek.com/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "kimi": "https://api.moonshot.cn/v1",
}


def _builtin_guess(model, key):
    """Best-effort guess for public providers; returns None for anything
    custom (internal gateways, vendor-suffixed names, empty hints)."""
    if key:
        for prefix, upstream in BUILTIN_KEY_PREFIXES.items():
            if key.startswith(prefix):
                return upstream
    if model:
        low = model.lower()
        for hint, upstream in BUILTIN_MODEL_HINTS.items():
            if hint in low:
                return upstream
    return None


def _read_zcode(path=None):
    """~/.zcode/v2/config.json -> provider map."""
    p = path or os.path.expanduser("~/.zcode/v2/config.json")
    try:
        d = json.load(open(p))
    except Exception:
        return {}
    routes = {}
    for _, prov in (d.get("provider") or {}).items():
        base = (prov.get("options") or {}).get("baseURL")
        key = (prov.get("options") or {}).get("apiKey", "")
        if not base:
            continue
        for model in (prov.get("models") or {}):
            routes[model] = (base, key)
    return routes


def _read_codex(path=None):
    """~/.codex/config.toml -> model_providers."""
    p = path or os.path.expanduser("~/.codex/config.toml")
    try:
        text = open(p).read()
    except Exception:
        return {}
    routes = {}
    try:
        import tomllib  # 3.11+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            return {}
    try:
        d = tomllib.loads(text)
    except Exception:
        return {}
    for _, prov in (d.get("model_providers") or {}).items():
        base = prov.get("base_url")
        key = os.environ.get(prov.get("env_key", ""), "")
        if not base:
            continue
        # codex providers don't enumerate models; register under provider name
        routes[prov.get("name", "")] = (base, key)
    return routes


def _read_hermes(path=None):
    """~/.hermes/models.json (list) + ~/.hermes/config.yaml fallback."""
    p = path or os.path.expanduser("~/.hermes/models.json")
    try:
        d = json.load(open(p))
    except Exception:
        return {}
    routes = {}
    for entry in (d if isinstance(d, list) else []):
        name = entry.get("name") or entry.get("model")
        base = entry.get("baseUrl") or entry.get("base_url")
        if name and base:
            routes[name] = (base, "")
    return routes


CLIENT_READERS = {"zcode": _read_zcode, "codex": _read_codex,
                  "hermes": _read_hermes}


class RouteTable:
    """model -> (upstream_url, key). Three tiers: explicit > config > builtin."""

    def __init__(self, routes=None, routes_from_clients=None, use_builtin=True):
        self.routes = {}          # explicit --route entries
        self.config_routes = {}   # read from client configs
        self.use_builtin = use_builtin
        for name in (routes_from_clients or []):
            reader = CLIENT_READERS.get(name)
            if reader:
                self.config_routes.update(reader())
        for entry in (routes or []):
            self.add_route(entry)

    def add_route(self, spec):
        """--route spec: 'model=url[:key]' (key optional).

        url may contain '://' so we split on the LAST ':' only when what
        follows looks like a key (contains no '/')."""
        name, _, rest = spec.partition("=")
        name, rest = name.strip(), rest.strip()
        if not name or not rest:
            return
        url, key = rest, ""
        head, sep, tail = rest.rpartition(":")
        if sep and "/" not in tail and head.startswith(("http://", "https://")):
            url, key = head, tail
        self.routes[name] = (url, key)

    def resolve(self, model, client_key=""):
        """Returns (upstream, key) or raises KeyError with a helpful hint."""
        if model in self.routes:
            url, key = self.routes[model]
            return url, (key or client_key or "")
        if model in self.config_routes:
            url, key = self.config_routes[model]
            return url, (key or client_key or "")
        if self.use_builtin:
            guessed = _builtin_guess(model, client_key)
            if guessed:
                return guessed, client_key
        raise KeyError(model)
