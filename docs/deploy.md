# Releasing and hosting

[First release](#first-release) lists the one-time setup on PyPI, Zenodo and Hugging Face. The rest of this page is about hosting the explorer.

## The container

One container runs what `qubitlab serve` runs: the API and the built explorer on one port. The `Dockerfile` at the root builds it.

```bash
docker build -t qubitlab .
docker run -p 7860:7860 qubitlab     # then open http://localhost:7860
```

The build precomputes the slow pages (`python -m qubitlab.warm`): every view's defaults, the published presets, the error-correction curves and every check. The first visitor does not wait for them.

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

## First release

One-time setup, about an hour, all in the browser. Do steps 1 and 2 before pushing the tag; step 4 can come any time.

1. **PyPI** (so `pipx run qubitlab serve` works). On pypi.org, sign in, then Account settings → Publishing → *Add a new pending publisher* → GitHub, with:
   PyPI project name `qubitlab`, owner `pkarakala`, repository `qubit-lab`, workflow `release.yml`, environment `pypi`.
   No token is stored anywhere: PyPI trusts the workflow run.
2. **Zenodo** (a DOI for citing a release). On zenodo.org, log in with GitHub, open the account menu → GitHub, press *Sync now*, and switch on `pkarakala/qubit-lab`. Every GitHub release from then on is archived with a DOI, using CITATION.cff for the authors and title.
3. **Tag the release.** With the version in pyproject.toml and its CHANGELOG.md entry on main:
   ```bash
   git checkout main && git pull
   git tag v1.0.0 && git push origin v1.0.0
   ```
   The release workflow builds the explorer into the wheel, smoke-tests `qubitlab serve`, publishes to PyPI and makes the GitHub release, which Zenodo then archives.
4. **Hugging Face Space** (a public link anyone can open): the steps in [Hugging Face Spaces](#hugging-face-spaces) above.
5. **Afterwards.** Copy the DOI badge from the Zenodo record into README.md, and the DOI into CITATION.cff (`doi:`).
