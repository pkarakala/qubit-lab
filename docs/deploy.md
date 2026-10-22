# Hosting the explorer

One container serves the API and the built explorer on one port. The `Dockerfile` at the root builds it.

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
| `QUBITLAB_RATE` | 60/minute | Requests per visitor (by IP, through the proxy). Over it: HTTP 429. |
| `QUBITLAB_STATIC` | web/dist | Serve the explorer from the API process. |
| `QUBITLAB_CACHE` | .cache/qubitlab | Where results are cached on disk. |

Every request is also bounded by its own input limits (grid size, number of steps, durations), so one request cannot run for minutes.

## Hugging Face Spaces

1. On huggingface.co, create a Space: SDK **Docker**, hardware **CPU basic** (free).
2. Push this repository to the Space's git remote. The Space's `README.md` must start with this header:

   ```yaml
   ---
   title: Qubit Lab
   emoji: 🐈
   colorFrom: indigo
   colorTo: gray
   sdk: docker
   app_port: 7860
   license: apache-2.0
   ---
   ```

3. The Space builds the Dockerfile and goes live at `https://huggingface.co/spaces/<user>/qubit-lab`.

A free Space sleeps after 48 hours without visitors and wakes on the next visit.
