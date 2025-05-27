# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

# Используем Python 3.12 для лучшей совместимости с библиотеками
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Set environment variables for matplotlib and nltk
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV NLTK_DATA=/usr/local/nltk_data

WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Download NLTK data as root user before switching to appuser
RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data'); nltk.download('stopwords', download_dir='/usr/local/nltk_data')" || echo "NLTK download failed, continuing..."

# Create directories with proper permissions
RUN mkdir -p /tmp/matplotlib /usr/local/nltk_data && \
    chmod 755 /tmp/matplotlib /usr/local/nltk_data

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Give appuser access to necessary directories
RUN chown -R appuser:appuser /app /tmp/matplotlib && \
    chmod -R 755 /usr/local/nltk_data

# Copy the source code into the container.
COPY . .

# Make sure appuser owns the app directory
RUN chown -R appuser:appuser /app

# Switch to the non-privileged user to run the application.
USER appuser

# Change working directory to where the models and code are located
WORKDIR /app/.venv/Lib

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application from the correct directory where models/ folder exists
CMD python simple_server.py --port 8000
