# Base image
FROM python:3.10-slim


# Install required system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
#COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Change the working directory to the `app` directory
WORKDIR /app

# Copy the lockfile and `pyproject.toml` into the image
ADD uv.lock /app/uv.lock
ADD pyproject.toml /app/pyproject.toml

# Install dependencies
RUN uv sync --frozen --no-install-project

# Copy the project into the image
ADD . /app

# Sync the project
RUN uv sync --frozen

# Default command to run the `train` script
CMD ["uv", "run", "train"]
