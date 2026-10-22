"""Command line: `qubitlab checks` runs theory vs simulation, `qubitlab serve` the explorer."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="qubitlab")
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("checks", help="compare every formula with the simulation")
    c.add_argument("--fast", action="store_true", help="skip Liouvillian spectra")
    s = sub.add_parser("serve", help="run the explorer and its API on one port")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()

    if args.cmd == "checks":
        from qubitlab.checks import run_all

        results = run_all(include_slow=not args.fast)
        for r in results:
            mark = "ok " if r.as_expected else "!! "
            print(f"{mark}{r.verdict.value:8} {r.key:45} err {r.rel_error:9.2e}  {r.regime}")
        surprises = sum(not r.as_expected for r in results)
        print(f"\n{len(results)} checks, {surprises} surprises")
        raise SystemExit(1 if surprises else 0)

    import threading
    import webbrowser

    import uvicorn

    from qubitlab.serve import app_with_explorer, has_explorer

    if not has_explorer():
        print("No built explorer here, so this serves the API only.")
        print("From a git checkout: cd web && npm run build:package, or use ./scripts/dev.sh.")
        uvicorn.run("qubitlab.api:app", port=args.port)
        return
    url = f"http://127.0.0.1:{args.port}"
    print(f"Qubit Lab: {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app_with_explorer(), port=args.port)
