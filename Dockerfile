FROM python:3.10-slim

WORKDIR /app

# 1) Install UV itself into the system environment
RUN pip install --no-cache-dir uv

# 2) Instruct UV to use this Python instead of vendoring one into /root
ENV UV_SYSTEM_PYTHON=1

# 3) Sync dependencies (no project code)
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-install-project --no-dev

# 4) Copy your app and install it
COPY . /app
RUN uv sync --frozen --no-dev

# 5) Use whatever UV/venv created on PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501
ENTRYPOINT ["streamlit","run","src/image_caption/scripts/app.py","--server.port=8501","--server.address=0.0.0.0"]
