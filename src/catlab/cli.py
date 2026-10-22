"""Command line: `catlab checks` prints theory vs simulation; `catlab serve` runs the API."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="catlab")
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("checks", help="compare every formula with the simulation")
    c.add_argument("--fast", action="store_true", help="skip Liouvillian spectra")
    s = sub.add_parser("serve", help="run the API for the explorer")
    s.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.cmd == "checks":
        from catlab.checks import run_all

        results = run_all(include_slow=not args.fast)
        for r in results:
            mark = "ok " if r.as_expected else "!! "
            print(f"{mark}{r.verdict.value:8} {r.key:45} err {r.rel_error:9.2e}  {r.regime}")
        surprises = sum(not r.as_expected for r in results)
        print(f"\n{len(results)} checks, {surprises} surprises")
        raise SystemExit(1 if surprises else 0)

    import uvicorn

    uvicorn.run("catlab.api:app", port=args.port)
