FROM python:3.10.14-slim-bullseye

# Set HOME so Streamlit and Transformers cache paths are in a writable location
ENV HOME=/app
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/hub

WORKDIR /app

# Install system dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
 && rm -rf /var/lib/apt/lists/*

# Create and set permissions for cache and config directories
RUN mkdir -p $HOME/.streamlit $TRANSFORMERS_CACHE \
 && chmod -R a+rwx $HOME/.streamlit $TRANSFORMERS_CACHE

# Copy dependency specs and project metadata
COPY requirements.txt pyproject.toml /app/

# Install Python dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project source and install as editable package
COPY . /app
RUN pip install --no-cache-dir -e .

# Expose port and add healthcheck
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit app
ENTRYPOINT ["streamlit", "run", "src/image_caption/scripts/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
