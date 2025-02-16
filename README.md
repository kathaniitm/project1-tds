# Automation Agent for DataWorks Solutions

This repository contains an automation agent built for DataWorks Solutions. The agent processes a wide variety of tasks—ranging from log analysis, code formatting, data scraping, and media processing—to generate actionable insights. It integrates multiple steps and leverages a Large Language Model (LLM) (via AIRPROXY) to parse plain-English instructions and dispatch them to appropriate internal functions.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Available Operations](#available-operations)
  - [A Operations](#a-operations)
  - [B Operations](#b-operations)
  - [Additional Tasks](#additional-tasks)
- [Docker Setup](#docker-setup)
- [License](#license)

## Features

- **Multi-Step Automation:** Parse plain-English tasks using an LLM and execute multi-step workflows.
- **Modular Operations:**
  - **A Operations:** Tasks like data generation, formatting, date processing, contact sorting, log analysis, email extraction, credit card extraction, comment similarity (via embeddings), and SQLite query.
  - **B Operations:** More complex operations such as fetching API data, Git repository automation, running SQL queries on databases, website scraping, image processing, audio transcription, Markdown to HTML conversion, and CSV-to-JSON conversion.
- **LLM Parameter Extraction:** Uses a master prompt along with user input to extract parameters for each operation.
- **Environment Variables:** Sensitive information (e.g., `AIPROXY_TOKEN`) is loaded via environment files.

## Project Structure

```
project1-tds/
├── Dockerfile
├── main.py                # Main Flask application entry point
├── requirements.txt       # Python dependencies
├── .env                   # Environment file (not committed)
├── README.md              # This file
└── (other modules and helper files)
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/project1-tds.git
   cd project1-tds
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**

   Create a file named **.env** in the project root with your sensitive values. For example:

   ```
   AIPROXY_TOKEN=your_actual_token_here
   ```

   The application loads these using `python-dotenv`.

## Usage

To run the automation agent locally, simply execute:

```bash
python main.py
```

The agent exposes two main endpoints:

- **Run Task:**  
  `http://localhost:8000/run?task=<your_task_description>`  
  This endpoint accepts a plain-English task (e.g., “Format my markdown file…”). The agent uses an LLM to parse the instruction and executes the corresponding operation.

- **Read File:**  
  `http://localhost:8000/read?path=<file_path>`  
  This endpoint returns the exact contents of the file at `<file_path>` for verification.

## Available Operations

### A Operations

- **A1:** Install a package and run a remote data generation script with a user email.
- **A2:** Format a Markdown file using Prettier.
- **A3:** Process a date file by counting specific weekdays and writing the count to a file.
- **A4:** Sort contacts in a JSON file by last name then first name.
- **A5:** Extract the first line from the 10 most recent log files.
- **A6:** Index Markdown documents by extracting the first H1 from each file.
- **A7:** Extract the sender’s email address from an email message using LLM.
- **A8:** Extract a credit card number from an image by sending its Base64 encoding to an LLM.
- **A9:** Use embeddings to find the most similar pair of comments in a file.
- **A10:** Query a SQLite database to calculate total ticket sales.

### B Operations

- **B3:** Fetch data from an API and save it.
- **B4:** Clone a Git repository and make an empty commit (with defaults) and print git status.
- **B5:** Run a SQL query on a default SQLite database (creates a sample DB if not exists) and return table names.
- **B6:** Scrape data from a website (default selector and output file if not provided).
- **B7:** Compress or resize an image (using defaults; accepts both local paths and URLs).
- **B8:** Transcribe audio from an MP3 file (downloads if URL provided) and save transcript.
- **B9:** Convert Markdown to HTML (and return the HTML string).
- **B10:** Convert a CSV file to JSON without filtering (using a default CSV URL if not provided).

### Additional Tasks (Examples)

- **Convert JSON to CSV:** Convert API JSON data to CSV.
- **Translate Text:** Use the LLM to translate text from English to another language.
- **Extract Metadata:** Download an image and extract its EXIF metadata.
- **Dynamic Password Generator:** Generate strong passwords personalized with the user's name.

## Docker Setup

### Dockerfile

The Dockerfile is located in the project root and looks like this:

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Accept AIPROXY_TOKEN as a build argument (optional)
ARG AIPROXY_TOKEN

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV AIPROXY_TOKEN=${AIPROXY_TOKEN}

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Optionally copy .env file (if you wish to provide defaults via .env)
COPY .env .env

# Copy the rest of the application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
```

### Build & Run Docker Image

1. **Build the Docker Image:**

   ```bash
   docker build -t your-dockerhub-username/repo-name .
   ```

2. **Run the Docker Container:**

   ```bash
   docker run --rm -e AIPROXY_TOKEN=$AIPROXY_TOKEN -p 8000:8000 your-dockerhub-username/repo-name
   ```

Your API endpoints will be available at:

- `http://localhost:8000/run?task=...`
- `http://localhost:8000/read?path=...`
