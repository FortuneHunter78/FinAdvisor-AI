FROM python:3.12

# Set working directory
WORKDIR /src

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements file
COPY requirements.txt src/requirements.txt

# Install system dependencies (OpenSSL development libraries and build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libssl-dev gcc make && \
    rm -rf /var/lib/apt/lists/*

# Explicitly install pyOpenSSL before installing other dependencies
RUN pip install --no-cache-dir --upgrade pyOpenSSL

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r src/requirements.txt --verbose

# Copy application code
COPY . .

# Expose port
EXPOSE 8089:80

# Run the application
CMD ["fastapi", "run", "main.py", "--port", "80"]