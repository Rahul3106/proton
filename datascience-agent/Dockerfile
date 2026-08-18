# PROTON Data Science Agent — Dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency specifications
COPY pyproject.toml README.md ./

# Install project dependencies into virtual environment
RUN uv venv && uv pip install -e .

# Copy source code and static web assets
COPY src ./src

# Create outputs directories
RUN mkdir -p outputs/plots outputs/reports outputs/models outputs/uploads

# Expose web dashboard port
EXPOSE 8000

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Launch FastAPI web server and dashboard
CMD ["uv", "run", "proton", "serve", "--host", "0.0.0.0", "--port", "8000"]
