# Base image
FROM python:3.11-slim

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent python buffering stdout
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (needed by some python packages)
RUN apt-get update \
    && apt-get install -y build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (better docker cache)
COPY requirements.txt .

# Install python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]