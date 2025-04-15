FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Debugging
RUN echo "Listing directory structure:"
RUN ls -la
RUN echo "Checking hotel_mapping directory:"
RUN ls -la hotel_mapping/
RUN echo "Python packages:"
RUN pip list


# Create models directory if it doesn't exist
RUN mkdir -p models
RUN mkdir -p trained_model

# The PORT environment variable is provided by Cloud Run
# We need to make sure our application listens on this port
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV DEBUG=True

RUN pip install gunicorn uvloop httptools


RUN echo '#!/bin/bash\n\
echo "Starting application..."\n\
echo "Current directory: $(pwd)"\n\
echo "PORT: $PORT"\n\
echo "Python path:"\n\
python -c "import sys; print(sys.path)"\n\
echo "Checking if app module can be imported:"\n\
python -c "from hotel_mapping.app import app; print(\"Import successful\")"\n\
echo "Starting server..."\n\
exec gunicorn hotel_mapping.app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --workers 2 --timeout 300 --log-level debug\n\
' > /app/start.sh && \
chmod +x /app/start.sh
# Run the application with Uvicorn
# Note: We're using the PORT environment variable here
CMD ["/app/start.sh"]
