# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to avoid .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application code
COPY . .

# Expose port 8000 (Flask app will run on this port)
EXPOSE 8000

# Run the application. Environment variables from .env will be loaded by python-dotenv in your code.
CMD ["python", "main.py"]

