FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory if it doesn't exist
RUN mkdir -p models
RUN mkdir -p trained_model

# The PORT environment variable is provided by Cloud Run
# We need to make sure our application listens on this port
ENV PORT=8080

# Run the application with Uvicorn
# Note: We're using the PORT environment variable here
CMD uvicorn hotel_mapping.app:app --host 0.0.0.0 --port ${PORT}
