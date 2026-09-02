# -*- coding: utf-8 -*-
"""CLI: `earcon serve` starts the learning proxy.

The only thing you configure here is the judge channel; the work channel
is pure pass-through (clients keep their own upstream, key and model).

Examples:
    earcon serve --judge-upstream https://api.deepseek.com/v1 \
        --judge-api-key $DSK --judge-model deepseek-chat

    # pure observation (no injection) while you assess judge quality:
    earcon serve --judge-... --no-inject
"""

import argparse
import json
import os


def build_parser():
    p = argparse.ArgumentParser(prog="earcon", description=__doc__)
    sub = p.add_subparsers(dest="cmd")
    serve = sub.add_parser("serve", help="run the learning proxy gateway")
    serve.add_argument("--upstream", default=os.environ.get("EARCON_UPSTREAM", ""),
                       help="fallback OpenAI-compatible upstream, used only when "
                            "the client doesn't carry one (env: EARCON_UPSTREAM). "
                            "Most clients already point at their real upstream "
                            "before switching baseURL - leave this empty")
    serve.add_argument("--api-key", default=os.environ.get("EARCON_API_KEY", ""),
                       help="fallback key for the work channel, used only when "
                            "the client didn't send one (env: EARCON_API_KEY)")
    serve.add_argument("--port", type=int, default=8800)
    serve.add_argument("--db", default="earcon_memory.db")
    serve.add_argument("--judge-model", required=True,
                       help="the ONLY required configuration: which model scores "
                            "sessions at close. Configured once here, it judges "
                            "every session regardless of which model does the work")
    serve.add_argument("--judge-upstream", required=True,
                       help="OpenAI-compatible base URL of the judge channel "
                            "(env: EARCON_JUDGE_UPSTREAM)")
    serve.add_argument("--judge-api-key", default=os.environ.get("EARCON_JUDGE_API_KEY", ""),
                       help="key for the judge channel (env: EARCON_JUDGE_API_KEY)")
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
    serve.add_argument("--route", action="append", default=[],
                       help="explicit model route 'model=url[:key]'; wins over "
                            "client configs and the built-in table (repeatable)")
    serve.add_argument("--routes-from", default="",
                       help="comma-separated clients to read routes from: "
                            "zcode,codex,hermes (reads their config files)")
    serve.add_argument("--no-builtin-routes", action="store_true",
                       help="disable the built-in public-cloud route table")
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
    # set before Gateway reads it for route self-filtering
    os.environ["EARCON_PORT"] = str(args.port)
    gw = Gateway(judge_model=args.judge_model, db_path=args.db,
                 judge_upstream=args.judge_upstream,
                 judge_api_key=args.judge_api_key,
                 upstream=args.upstream or None, api_key=args.api_key or None,
                 routes=args.route,
                 routes_from_clients=[s for s in args.routes_from.split(",") if s],
                 use_builtin_routes=not args.no_builtin_routes,
                 inject=not args.no_inject, extra_body=extra,
                 config={"session_timeout": args.session_timeout,
                         "inject_top_k": args.inject_top_k})
    app = create_app(gw)

    import uvicorn
    print("earcon gateway: http://127.0.0.1:%d" % args.port)
    n = len(gw.route_table.routes) + len(gw.route_table.config_routes)
    print("work channel: routed by model (%d routes loaded, builtin=%s)"
          % (n, "on" if gw.route_table.use_builtin else "off"))
    print("judge channel: %s -> %s" % (args.judge_model, args.judge_upstream))
    print("memory: %s | mode: %s" % (args.db,
                                     "record+inject" if gw.inject else "record-only"))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
