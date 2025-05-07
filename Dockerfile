# Use the UV image (includes uv CLI + vendored Python)
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /app

# Keep bytecode compilation and copy mode
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# 1) Sync only dependencies (no project code)
COPY uv.lock pyproject.toml /app/
RUN uv sync --frozen --no-install-project --no-dev

# 2) Copy your full source & install your project (no dev deps)
COPY . /app
RUN uv sync --frozen --no-dev

# 3) Fix permissions on UV’s Python runtime so it can load libpython3.x
RUN chmod -R a+rX /root/.local/share/uv/python

# Ensure your venv’s Python and scripts are on the PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501

# Launch your Streamlit app
ENTRYPOINT ["streamlit","run","src/image_caption/scripts/app.py","--server.port=8501","--server.address=0.0.0.0"]

