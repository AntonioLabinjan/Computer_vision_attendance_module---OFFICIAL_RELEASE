# Use Python slim image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy application files (ensure to include only necessary files using .dockerignore)
COPY . /app

# Create known_faces directory (if not already created by COPY)
RUN mkdir -p /app/known_faces

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies with increased timeout # ovaj timeout dignat na nešto divlje
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt 

# Test OpenCV installation
RUN python -c "import cv2; print(cv2.__version__)"

# Expose the Flask app port
EXPOSE 5000

# Set the command to run the Flask application
CMD ["python", "app.py"]
