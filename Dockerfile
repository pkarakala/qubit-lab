# Qubit Lab in one container: the explorer (built with Node) served by the FastAPI app.
# Hugging Face Spaces runs it as user 1000 on port 7860; see docs/deploy.md.

FROM node:22.22.0-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --legacy-peer-deps
COPY web/ ./
RUN npm run build

FROM python:3.12.12-slim
RUN pip install --no-cache-dir uv==0.11.32
RUN useradd --create-home --uid 1000 lab
USER lab
WORKDIR /home/lab/app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/home/lab/app/.venv/bin:$PATH" \
    QUBITLAB_CACHE=/home/lab/app/.cache/qubitlab \
    QUBITLAB_STATIC=/home/lab/app/web/dist \
    QUBITLAB_SOLVES=2 \
    QUBITLAB_RATE=60/minute
COPY --chown=lab pyproject.toml uv.lock README.md references.bib ./
COPY --chown=lab src/ src/
COPY --chown=lab docs/hand_calcs/ docs/hand_calcs/
RUN uv sync --frozen --no-dev
COPY --from=web --chown=lab /web/dist web/dist
# Precompute the slow pages (all checks, the QEC curves) so the first visitor does not wait.
COPY --chown=lab scripts/warm_cache.py scripts/
RUN python scripts/warm_cache.py
EXPOSE 7860
CMD ["uvicorn", "qubitlab.api:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
