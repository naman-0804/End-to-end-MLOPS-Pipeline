# 1. Use an official lightweight Python runtime
FROM python:3.11-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Set working directory inside container
WORKDIR /app

# 4. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy necessary application code and models
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

# 6. Expose the port FastAPI runs on
EXPOSE 8000

# 7. Command to run the application using Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
