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

# Expose port (FastAPI 默认使用 8000 端口)
EXPOSE 8000

# Run the application with Uvicorn
CMD ["uvicorn", "hotel_mapping.app:app", "--host", "0.0.0.0", "--port", "8000"]
