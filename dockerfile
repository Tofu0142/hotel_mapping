FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    # Add dependencies for matplotlib
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models
RUN mkdir -p trained_model

# Install the application
RUN pip install -e .

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create a startup script with debugging information
RUN echo '#!/bin/bash\n\
echo "Starting application..."\n\
echo "Installed packages:"\n\
pip list\n\
\n\
echo "Starting server on port $PORT..."\n\
exec gunicorn --bind 0.0.0.0:$PORT \\\n\
    --workers 1 \\\n\
    --worker-class uvicorn.workers.UvicornWorker \\\n\
    --timeout 300 \\\n\
    --log-level info \\\n\
    hotel_mapping.app:app\n\
' > /app/start.sh && \
chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"]
