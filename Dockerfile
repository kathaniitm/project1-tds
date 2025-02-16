FROM python:3.10-slim

# Install system dependencies:
# - git for GitPython.
# - ffmpeg for audio processing.
# - curl, gnupg, and build-essential for Node.js installation and building packages.
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    curl \
    gnupg \
    build-essential \
  && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
  && apt-get install -y nodejs \
  && rm -rf /var/lib/apt/lists/*

# Set environment variables for Python.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container.
WORKDIR /app

# Copy requirements.txt and install Python dependencies.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose the port on which your Flask app will run.
EXPOSE 8000

# Run the application.
CMD ["python", "main.py"]






