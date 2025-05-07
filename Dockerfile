FROM python:3.10-slim


COPY --from=ghcr.io/astral-sh/uv:python3.10-bookworm-slim /uv /bin/uv

WORKDIR /app


ENV XDG_CACHE_HOME=/app/.cache


COPY pyproject.toml uv.lock /app/


RUN --mount=type=cache,target=/app/.cache \
    uv sync --frozen --no-install-project --no-dev


COPY . /app
RUN --mount=type=cache,target=/app/.cache \
    uv sync --frozen --no-dev


ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501
ENTRYPOINT ["streamlit","run","src/image_caption/scripts/app.py","--server.port=8501","--server.address=0.0.0.0"]
