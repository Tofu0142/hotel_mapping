FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .

# Install specific versions to resolve compatibility issues
# The cached_download function exists in huggingface_hub 0.12.1
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir transformers==4.26.0 && \
    pip install --no-cache-dir huggingface_hub==0.12.1 && \
    pip install --no-cache-dir sentence-transformers==2.2.2 && \
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
echo "Python packages:"\n\
pip list | grep -E "huggingface|sentence|transformers"\n\
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
