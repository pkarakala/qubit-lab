# Hosting the explorer

One container runs what `qubitlab serve` runs: the API and the built explorer on one port. The `Dockerfile` at the root builds it.

```bash
docker build -t qubitlab .
docker run -p 7860:7860 qubitlab     # then open http://localhost:7860
```

The build precomputes the slowest pages (`scripts/warm_cache.py`): every check, the error-correction curves, and the cat vs transmon table. The first visitor does not wait for them.

## Limits for a public server

All off unless set. The Dockerfile sets them.

| Variable | Dockerfile value | What it does |
| --- | --- | --- |
| `QUBITLAB_SOLVES` | 2 | Most simulations running at once. The rest queue. |
| `QUBITLAB_RATE` | 120/minute | Requests per visitor (by IP, through the proxy). Over it: HTTP 429. |
| `QUBITLAB_CACHE` | .cache/qubitlab | Where results are cached on disk. |

Every request is also bounded by its own input limits (grid size, number of steps, durations), so one request cannot run for minutes.

## Hugging Face Spaces

One-time setup, about five minutes:

1. On huggingface.co, create a Space named `qubit-lab`: SDK **Docker**, hardware **CPU basic** (free).
2. In your Hugging Face settings, create an access token with **write** access.
3. In the GitHub repo, Settings → Secrets and variables → Actions:
   - secret `HF_TOKEN`: that token;
   - variable `HF_SPACE`: `<your-hf-username>/qubit-lab`.
4. Run the **space** workflow once (Actions → space → Run workflow), or push to main.

From then on, every push to main redeploys. The workflow (`.github/workflows/space.yml`) adds the Space's settings header to README.md and leaves out the screenshots, which the Space would refuse as binary files. The Space builds the Dockerfile and goes live at `https://huggingface.co/spaces/<your-hf-username>/qubit-lab`.

A free Space sleeps after 48 hours without visitors and wakes on the next visit, which takes about a minute.

To check a deployed copy end to end: `cd web && QUBITLAB_URL=https://<your-hf-username>-qubit-lab.hf.space npx playwright test`.
