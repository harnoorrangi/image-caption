FROM python:3.10-slim

WORKDIR /app

# Install UV and your other dependencies
RUN pip install --no-cache-dir uv

# Copy & sync your project
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-install-project --no-dev
COPY . /app
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "src/image_caption/scripts/app.py","--server.port=8501","--server.address=0.0.0.0"]
