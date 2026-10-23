"""One process for everything: the API and the built explorer on one port.

`npm run build:package` (in web/) writes the explorer to src/qubitlab/explorer/, and the
release build ships it inside the wheel. Then `pip install qubitlab && qubitlab serve` is all a
user needs: no Node, no second server.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

EXPLORER = Path(__file__).resolve().parent / "explorer"


def has_explorer(explorer: Path = EXPLORER) -> bool:
    return (explorer / "index.html").is_file()


def app_with_explorer(explorer: Path = EXPLORER) -> FastAPI:
    """The API routes, plus the explorer's static files at /."""
    from qubitlab import __version__
    from qubitlab.api import add_rate_limit, lifespan
    from qubitlab.api import app as api

    if not has_explorer(explorer):
        raise FileNotFoundError(
            f"no built explorer in {explorer}; run `npm run build:package` in web/"
        )
    root = FastAPI(
        title="qubitlab", version=__version__, docs_url=None, redoc_url=None, lifespan=lifespan
    )
    add_rate_limit(root)  # the hosting limits in qubitlab.api, when set
    # The API's own route objects, not include_router: slowapi only sees plain routes.
    root.router.routes.extend(r for r in api.routes if getattr(r, "path", "").startswith("/api/"))
    root.mount("/", StaticFiles(directory=explorer, html=True), name="explorer")
    return root
