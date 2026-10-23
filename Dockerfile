# Qubit Lab in one container: `qubitlab serve`'s app (API + built explorer) behind uvicorn.
# Hugging Face Spaces runs it as user 1000 on port 7860; see docs/deploy.md.

FROM node:22.22.0-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --legacy-peer-deps
COPY web/ ./
RUN npm run build:package

FROM python:3.12.12-slim
RUN pip install --no-cache-dir uv==0.11.32
RUN useradd --create-home --uid 1000 lab
USER lab
WORKDIR /home/lab/app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/home/lab/app/.venv/bin:$PATH" \
    QUBITLAB_CACHE=/home/lab/app/.cache/qubitlab \
    QUBITLAB_SOLVES=2 \
    QUBITLAB_RATE=120/minute
COPY --chown=lab pyproject.toml uv.lock README.md references.bib ./
COPY --chown=lab src/ src/
COPY --chown=lab docs/hand_calcs/ docs/hand_calcs/
RUN uv sync --frozen --no-dev
COPY --from=web --chown=lab /src/qubitlab/explorer src/qubitlab/explorer
# Precompute the slow pages (all checks, the QEC curves) so the first visitor does not wait.
RUN python -m qubitlab.warm
EXPOSE 7860
CMD ["uvicorn", "--factory", "qubitlab.serve:app_with_explorer", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
