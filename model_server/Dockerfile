# ========================
# Dockerfile for Model Server
# ========================

# Base image with Python 3.11
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn[standard] torchvision pillow opencv-python

# Copy the entire model server code
COPY . .

# Expose the port used by FastAPI
EXPOSE 8000

# Command to run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
