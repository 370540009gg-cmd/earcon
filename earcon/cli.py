# -*- coding: utf-8 -*-
"""CLI: `earcon serve` starts the learning proxy.

Examples:
    earcon serve --upstream https://api.openai.com/v1 --api-key $KEY \
        --judge-model gpt-4o-mini --port 8800

    # pure observation (no injection) while you assess judge quality:
    earcon serve --upstream ... --no-inject
"""

import argparse
import json
import os


def build_parser():
    p = argparse.ArgumentParser(prog="earcon", description=__doc__)
    sub = p.add_subparsers(dest="cmd")
    serve = sub.add_parser("serve", help="run the learning proxy gateway")
    serve.add_argument("--upstream", required=True,
                       help="OpenAI-compatible upstream base URL, e.g. "
                            "https://api.openai.com/v1 "
                            "(env: EARCON_UPSTREAM)")
    serve.add_argument("--api-key", default=os.environ.get("EARCON_API_KEY", ""),
                       help="key used for upstream calls; clients' own keys "
                            "are ignored (env: EARCON_API_KEY)")
    serve.add_argument("--port", type=int, default=8800)
    serve.add_argument("--db", default="earcon_memory.db")
    serve.add_argument("--judge-model", required=True,
                       help="model used for credit assignment")
    serve.add_argument("--no-inject", action="store_true",
                       help="record-only mode: judge sessions, never inject")
    serve.add_argument("--judge-extra-body", default=None,
                       help="JSON object merged into judge requests, for "
                            "vendor-private params (e.g. disabling a "
                            "reasoning mode). Never sent to the main path.")
    serve.add_argument("--session-timeout", type=float, default=1800,
                       help="seconds of inactivity before a session is "
                            "closed and judged (default 1800)")
    serve.add_argument("--inject-top-k", type=int, default=5)
    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    if args.cmd != "serve":
        p.print_help()
        return 2

    extra = json.loads(args.judge_extra_body) if args.judge_extra_body else None

    # local imports so the core library never requires fastapi
    from earcon.gateway import Gateway, create_app
    gw = Gateway(upstream=args.upstream, api_key=args.api_key,
                 db_path=args.db, judge_model=args.judge_model,
                 inject=not args.no_inject, extra_body=extra,
                 config={"session_timeout": args.session_timeout,
                         "inject_top_k": args.inject_top_k})
    app = create_app(gw)

    import uvicorn
    print("earcon gateway: http://127.0.0.1:%d  ->  %s" % (args.port, args.upstream))
    print("memory: %s | mode: %s" % (args.db,
                                     "record+inject" if gw.inject else "record-only"))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
