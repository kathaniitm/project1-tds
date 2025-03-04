import os
import random
import re
import json
import glob
import sqlite3
import hashlib
import subprocess
import urllib.request
import requests
import ssl
import base64
from git import Repo, GitCommandError
from datetime import datetime
from flask import Flask, request, jsonify, make_response
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from pydub import AudioSegment
import speech_recognition as sr
import markdown
import pandas as pd
import numpy as np
import imageio_ffmpeg as ffmpeg
from dotenv import load_dotenv
import string
import sqlite3
import duckdb





AudioSegment.converter = ffmpeg.get_ffmpeg_exe()
AudioSegment.ffprobe = ffmpeg.get_ffmpeg_exe()  # Using ffmpeg binary as a substitute for ffprobe.


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

ARTIFACTS_DIR = "artifacts"
DATA_DIR = "data"

# Ensure necessary folders exist on startup.
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

###############################################
# AIRPROXY Configuration
###############################################
load_dotenv()  # Loads environment variables from .env

AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
AIPROXY_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

###############################################
# Operation Functions (A1 - A10)
###############################################

def op_a1(task: str) -> str:
   
    match = re.search(r'[\w\.-]+@[\w\.-]+', task)
    user_email = match.group(0) if match else "default@example.com"
    # Define project root and data directory
    project_root = os.getcwd()
    DATA_DIR = os.path.join(project_root, "data")

    # User email to be passed as argument
    user_email = "22f3002542@ds.study.iitm.ac.in"  # Update if necessary

    try:
        subprocess.run(["pip", "install", "uv"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        pass

    datagen_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    datagen_script_path = os.path.join(project_root, "datagen.py")
    try:
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(datagen_url, context=context) as response:
            content = response.read()
        with open(datagen_script_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise Exception("Failed to download datagen.py: " + str(e))
    
    command = ["uv", "run", "datagen.py", user_email, "--root", "./data"]

    # Execute the command with the current working directory set to the project root
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_root
        )
    except Exception as e:
        raise Exception("Failed to execute datagen.py with uv: " + str(e))

    return f"Executed datagen.py with email {user_email}"

def op_a2(task: str) -> str:
   
    file_path = os.path.join(DATA_DIR, "format.md")
    if not os.path.exists(file_path):
        raise Exception(f"{file_path} does not exist.")

    # Check if Prettier is available by trying to get its version.
    try:
        result = subprocess.run(
            ["prettier", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Optionally, check that the version is 3.4.2. If not, reinstall.
        installed_version = result.stdout.strip()
        if installed_version != "3.4.2":
            raise Exception("Incorrect version")
    except Exception:
        # Prettier not found or not the correct version; install it globally.
        try:
            install_command = ["npm", "install", "-g", "prettier@3.4.2"]
            subprocess.run(
                install_command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise Exception("Failed to install Prettier: " + e.stderr)

    # Now run Prettier to format the file.
    command = ["prettier", "--write", file_path]
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except subprocess.CalledProcessError as e:
        raise Exception("Failed to format file using Prettier: " + e.stderr)
    
    return f"Formatted {file_path} using prettier@3.4.2"

def op_a3(task: str) -> str:
   
    input_path = os.path.join(DATA_DIR, "dates.txt")
    output_path = os.path.join(DATA_DIR, "dates-wednesdays.txt")
    if not os.path.exists(input_path):
        raise Exception(f"{input_path} does not exist.")
    
    # List of possible date formats
    date_formats = [
        "%Y-%m-%d",         # e.g., "2019-03-23"
        "%b %d, %Y",        # e.g., "Apr 21, 2005"
        "%Y/%m/%d %H:%M:%S", # e.g., "2003/03/18 21:45:29"
        "%d-%b-%Y"          # e.g., "07-Oct-2014"
    ]
    
    count = 0
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed_date = None
            # Try parsing using each format until one works
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(line, fmt)
                    break
                except Exception:
                    continue
            # If a valid date is parsed, check if it's a Wednesday (weekday() returns 2 for Wednesday)
            if parsed_date and parsed_date.weekday() == 2:
                count += 1
    
    with open(output_path, "w") as f:
        f.write(str(count))
    
    return f"Counted Wednesdays and wrote {count} to {output_path}"

def op_a4(task: str) -> str:
    
    input_path = os.path.join(DATA_DIR, "contacts.json")
    output_path = os.path.join(DATA_DIR, "contacts-sorted.json")
    if not os.path.exists(input_path):
        raise Exception(f"{input_path} does not exist.")

    with open(input_path, "r") as f:
        contacts = json.load(f)
    
    contacts_sorted = sorted(
        contacts,
        key=lambda x: (x.get("last_name", ""), x.get("first_name", ""))
    )

    with open(output_path, "w") as f:
        json.dump(contacts_sorted, f, indent=2)
    
    return f"Sorted contacts and wrote output to {output_path}"

def op_a5(task: str) -> str:
    
    logs_dir = os.path.join(DATA_DIR, "logs")
    output_path = os.path.join(DATA_DIR, "logs-recent.txt")
    log_files = glob.glob(os.path.join(logs_dir, "*.log"))
    if not log_files:
        raise Exception("No .log files found in " + logs_dir)
    
    log_files_sorted = sorted(log_files, key=os.path.getmtime, reverse=True)[:10]
    lines = []
    for log_file in log_files_sorted:
        with open(log_file, "r") as f:
            first_line = f.readline().strip()
            lines.append(first_line)
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    return f"Wrote the first lines of 10 most recent logs to {output_path}"

def op_a6(task: str) -> str:
    
    docs_dir = os.path.join(DATA_DIR, "docs")
    output_path = os.path.join(DATA_DIR, "docs", "index.json")
    if not os.path.exists(docs_dir):
        raise Exception(f"{docs_dir} does not exist.")
    
    index = {}
    for filepath in glob.glob(os.path.join(docs_dir, "**/*.md"), recursive=True):
        title = None
        with open(filepath, "r") as f:
            for line in f:
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
        rel_path = os.path.relpath(filepath, docs_dir)
        index[rel_path] = title if title else ""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(index, f, indent=2)
    
    return f"Created Markdown index at {output_path}"

def op_a7(task: str) -> str:
    
    input_path = os.path.join(DATA_DIR, "email.txt")
    output_path = os.path.join(DATA_DIR, "email-sender.txt")
    if not os.path.exists(input_path):
        raise Exception(f"{input_path} does not exist.")
    
    # Read the email message content.
    with open(input_path, "r") as f:
        content = f.read()
    
    # System prompt instructs the assistant to extract the sender's email address.
    system_prompt = (
        "You are an automation agent that extracts the sender's email address from a given email message. "
        "Given the email message text, return a JSON object with a single field 'email' that contains "
        "the sender's email address. For example, if the sender is 'sender@example.com', return: "
        "{\"email\": \"sender@example.com\"}. Do not include any additional text."
    )
    
    # Prepare the payload for AIRPROXY
    data = {
        "model": "gpt-4o-mini",  # adjust model name if needed
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
        "temperature": 0  # deterministic output
    }
    
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Send the request to AIRPROXY
    chat_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    response = requests.post(chat_url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception("AIRPROXY LLM failed: " + response.text)
    
    result = response.json()
    
    # Extract the assistant's reply and parse the JSON.
    try:
        assistant_content = result["choices"][0]["message"]["content"]
        parsed_content = json.loads(assistant_content)
        email_addr = parsed_content.get("email", "not_found@example.com")
    except Exception as e:
        raise Exception("Failed to parse LLM response: " + str(e))
    
    # Write the extracted email address to the output file.
    with open(output_path, "w") as f:
        f.write(email_addr)
    
    return f"Extracted sender email and wrote it to {output_path}"

def op_a8(task: str) -> str:
  
    input_path = os.path.join(DATA_DIR, "credit_card.png")
    output_path = os.path.join(DATA_DIR, "credit-card.txt")
    if not os.path.exists(input_path):
        raise Exception(f"{input_path} does not exist.")
    
    # Read and encode the image in Base64.
    try:
        with open(input_path, "rb") as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")
    except Exception as e:
        raise Exception("Failed to read and encode the image: " + str(e))
    
    # Construct the payload using the specified format.
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an automation agent that extracts a credit card number from an image. "
                    "When given an image (encoded as a data URL), return only a JSON object with a single field "
                    "'card_number' containing the extracted credit card number. For example: "
                    "{\"card_number\": \"4111111111111111\"}. Do not include any additional text."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the credit card number from this image."},
                    {
                        "type": "image_url",
                        "image_url": { "url": f"data:image/png;base64,{image_base64}" }
                    }
                ]
            }
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(AIPROXY_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception("AIRPROXY LLM failed for image: " + response.text)
    
    result = response.json()
    
    # Extract the assistant's reply from the chat-completion response.
    try:
        assistant_reply = result["choices"][0]["message"]["content"]
        # Expecting the assistant's reply to be a JSON string, e.g., {"card_number": "4111 1111 1111 1111"}
        parsed = json.loads(assistant_reply)
        card_number = parsed.get("card_number")
    except Exception as e:
        raise Exception("Failed to parse LLM response: " + str(e))
    
    if not card_number:
        raise Exception("No credit card number extracted by LLM.")
    
    # Remove any spaces from the card number.
    card_number = card_number.replace(" ", "")
    with open(output_path, "w") as f:
        f.write(card_number)
    
    return f"Extracted credit card number and wrote it to {output_path}"

def get_embedding(text: str) -> np.array:
    
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": "text-embedding-3-small"
    }
    response = requests.post("http://aiproxy.sanand.workers.dev/openai/v1/embeddings", headers=headers, json=data)
    if response.status_code != 200:
        raise Exception("Embedding API failed: " + response.text)
    
    result = response.json()
    # Expecting the API to return a JSON with an "embedding" field containing a list of numbers.
    print(result)
    embedding = result["data"][0]["embedding"]
    print(embedding)
    
    if embedding is None:
        raise Exception("No embedding returned for text: " + text)
    return np.array(embedding)

def cosine_similarity(vec1: np.array, vec2: np.array) -> float:
    
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def op_a9(task: str) -> str:
   
    input_path = os.path.join(DATA_DIR, "comments.txt")
    output_path = os.path.join(DATA_DIR, "comments-similar.txt")
    if not os.path.exists(input_path):
        raise Exception(f"{input_path} does not exist.")
    
    # Read comments from file.
    with open(input_path, "r") as f:
        comments = [line.strip() for line in f if line.strip()]
    
    if len(comments) < 2:
        raise Exception("Not enough comments to compare.")
    
    # Get embeddings for each comment using the AIRPROXY embeddings API.
    embeddings = []
    for comment in comments:
        embedding = get_embedding(comment)
        embeddings = embedding
    
    embeddings = np.array(embeddings)
    n = len(embeddings)
    
    # Compute the cosine similarity matrix.
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim  # symmetric matrix
    
    # Exclude self-similarity by setting the diagonal to -inf.
    np.fill_diagonal(sim_matrix, -np.inf)
    
    # Find the pair with the highest similarity.
    i, j = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
    similar_pair = (comments[i], comments[j])
    
    # Write the similar comments to the output file.
    with open(output_path, "w") as f:
        f.write(f"{similar_pair[0]}\n{similar_pair[1]}")
    
    return f"Found similar comments and wrote them to {output_path}"

def op_a10(task: str) -> str:
    
    db_path = os.path.join(DATA_DIR, "ticket-sales.db")
    output_path = os.path.join(DATA_DIR, "ticket-sales-gold.txt")
    if not os.path.exists(db_path):
        raise Exception(f"{db_path} does not exist.")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT units, price FROM tickets WHERE type = 'Gold'")
        rows = cursor.fetchall()
        print(rows)
        total_sales = sum(units * price for units, price in rows)
    except Exception as e:
        raise Exception("Database query failed: " + str(e))
    finally:
        conn.close()
    
    with open(output_path, "w") as f:
        f.write(str(total_sales))
    
    return f"Calculated total sales for 'Gold' tickets: {total_sales} and wrote to {output_path}"


def llm_extract_params(system_prompt: str, task: str) -> dict:
   
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ],
        "temperature": 0
    }
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    response = requests.post(AIPROXY_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception("AIRPROXY LLM failed: " + response.text)
    result = response.json()
    print(result)
    try:
        assistant_reply = result["choices"][0]["message"]["content"]
        params = json.loads(assistant_reply)
        return params
    except Exception as e:
        raise Exception("Failed to parse LLM extraction response: " + str(e))

def op_b3(task: str) -> str:
   
    system_prompt = (
        "Extract the following parameters from the task as JSON: "
        "{'api_url': <string>, 'output_file': <string>}."
        "For example: {\"api_url\": \"https://example.com/data\", \"output_file\": \"data/api_data.json\"}."
    )
    params = llm_extract_params(system_prompt, task)
    api_url = params.get("api_url") or "https://catfact.ninja/facts"
    output_file = params.get("output_file")
    if not api_url or not output_file:
        raise Exception("Missing parameters for B3 in the task.")
    
    # Fetch data from the API.
    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception("Failed to fetch data from API: " + response.text)
    
    # Ensure output file path is relative to project root.
    output_path = os.path.join(os.getcwd(), output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    
    return f"Fetched data from {api_url} and saved to {output_path}"

def op_b4(task: str) -> str:
   
    system_prompt = (
        "Extract the following parameters from the task as JSON (all parameters are optional): "
        "{'repo_url': <string, default 'https://github.com/kathaniitm/student-marks'>, "
        "'branch': <string, default 'main'>, "
        "'commit_message': <string, default 'Empty commit by automation agent'>}.\n"
        "For example: {\"repo_url\": \"https://github.com/kathaniitm/student-marks\", "
        "\"branch\": \"main\", \"commit_message\": \"Empty commit by automation agent\"}."
    )
    params = llm_extract_params(system_prompt, task)
    repo_url = params.get("repo_url") or 'https://github.com/kathaniitm/student-marks'
    print(repo_url)
    branch = params.get("branch") or "main"
    commit_message = params.get("commit_message") or "this is a test commit"
     # Clone repository into a temporary directory.
    clone_dir = os.path.join(os.getcwd(), "temp_repo")
    if os.path.exists(clone_dir):
        subprocess.run(["rm", "-rf", clone_dir])
    
    try:
        repo = Repo.clone_from(repo_url, clone_dir, branch=branch)
    except GitCommandError as e:
        raise Exception("Failed to clone repository: " + str(e))
    
    try:
        # Create an empty commit.
        repo.git.commit(allow_empty=True, m=commit_message)
    except GitCommandError as e:
        raise Exception("Failed to create empty commit: " + str(e))
    
    try:
        # Retrieve the git status after the commit.
        status = repo.git.status()
        print("Git status after commit:\n", status)
    except GitCommandError as e:
        raise Exception("Failed to get git status: " + str(e))

    # Clean up the clone directory.
    # subprocess.run(["rm", "-rf", clone_dir])
    
    return f"Cloned repo {repo_url} and created an empty commit with message '{commit_message}'"


def op_b6(task: str) -> str:
    
    system_prompt = (
            "Extract the following parameter from the task as JSON: {'url': <string>}."
            "For example: {\"url\": \"https://example.com\"}."
        )
    params = llm_extract_params(system_prompt, task)
    url = params.get("url") or "https://webscraper.io/"
    if not url:
        raise Exception("Missing 'url' parameter for B6 in the task.")
    
    # Default values if not provided
    selector = params.get("selector") or "body"
    output_file = params.get("output_file") or "data/scraped.json"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch website, change website url: " + response.text)
    
    soup = BeautifulSoup(response.text, "html.parser")
    elements = soup.select(selector)
    extracted = [el.get_text(strip=True) for el in elements]
    
    output_path = os.path.join(os.getcwd(), output_file)
    # Ensure the directory exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2)
    
    return f"Scraped data from {url} using selector '{selector}' and saved to {output_path}"

def op_b7(task: str) -> str:
   
    system_prompt = (
        "Extract the following parameters from the task as JSON (all parameters are optional): "
        "{'image_path': <string>, 'target_width': <int>, 'target_height': <int>, 'quality': <int>, 'output_file': <string>}."
        "For any parameter not provided, defaults will be used. return no value available as only single return parameter"
        "For example: {\"image_path\": \"https://example.com/image.jpg\", \"target_width\": 800, \"target_height\": 600, \"quality\": 85, \"output_file\": \"data/output.jpg\"}."
    )
    params = llm_extract_params(system_prompt, task)
    # Set defaults
    image_path = params.get("image_path") or "https://i.sstatic.net/l60Hf.png"
    target_width = int(params.get("target_width") or 800)
    target_height = int(params.get("target_height") or 600)
    quality = int(params.get("quality") or 85)
    output_file = params.get("output_file") or "data/output.jpg"

    # Determine if image_path is a URL or a local file path.
    if image_path.startswith("http"):
        # Download the image
        response = requests.get(image_path)
        if response.status_code != 200:
            raise Exception(f"Failed to download image from URL: {image_path}")
        image = Image.open(BytesIO(response.content))
    else:
        local_image_path = os.path.join(os.getcwd(), image_path)
        if not os.path.exists(local_image_path):
            raise Exception(f"Image file {local_image_path} does not exist.")
        image = Image.open(local_image_path)

    # Resize the image.
    image = image.resize((target_width, target_height))

    # Ensure the output directory exists.
    output_path = os.path.join(os.getcwd(), output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the image with specified quality.
    image.save(output_path, quality=quality)
    
    return f"Processed image saved to {output_path}"

def op_b8(task: str) -> str:
    """
    B8. Transcribe audio from an MP3 file.
    Optionally, the task may include:
      - audio_path: URL or local path to the MP3 file (default: a sample MP3 URL)
      - output_file: Path to save the transcript (default: "data/transcript.txt")
    
    If audio_path is a URL, the file is downloaded locally before processing.
    The function converts MP3 to WAV, transcribes using SpeechRecognition (Google API),
    and writes the transcript to the output file.
    This function installs and uses ffmpeg via the imageio-ffmpeg package, so that an external ffmpeg installation is not required.
    """
    system_prompt = (
        "Extract the following parameters from the task as JSON (both parameters are optional): "
        "{'audio_path': <string, default 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3'>, "
        "'output_file': <string, default 'data/transcript.txt'>}."
        "For example: {\"audio_path\": \"https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3\", "
        "\"output_file\": \"data/transcript.txt\"}."
    )
    params = llm_extract_params(system_prompt, task)
    # Default to a sample MP3 URL if not provided.
    audio_path = params.get("audio_path") or "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
    output_file = params.get("output_file") or "data/transcript.txt"
    
    # Determine local input path.
    if audio_path.startswith("http"):
        # Download the audio file.
        response = requests.get(audio_path)
        if response.status_code != 200:
            raise Exception(f"Failed to download audio from URL: {audio_path}")
        # Save to a temporary file in the data directory.
        local_audio_path = os.path.join(os.getcwd(), "data", "temp_audio.mp3")
        os.makedirs(os.path.dirname(local_audio_path), exist_ok=True)
        with open(local_audio_path, "wb") as f:
            f.write(response.content)
    else:
        local_audio_path = os.path.join(os.getcwd(), audio_path)
        if not os.path.exists(local_audio_path):
            raise Exception(f"Audio file {local_audio_path} does not exist.")
    
    # Convert MP3 to WAV using pydub.
    try:
        audio = AudioSegment.from_mp3(local_audio_path)
    except Exception as e:
        raise Exception("Failed to load MP3 file: " + str(e))
    
    wav_path = local_audio_path.rsplit(".", 1)[0] + ".wav"
    audio.export(wav_path, format="wav")
    
    # Transcribe the WAV file using SpeechRecognition.
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
    try:
        transcript = recognizer.recognize_google(audio_data)
    except Exception as e:
        raise Exception("Transcription failed: " + str(e))
    
    output_path = os.path.join(os.getcwd(), output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    # Clean up temporary files.
    os.remove(wav_path)
    if audio_path.startswith("http"):
        os.remove(local_audio_path)
    
    return f"Transcribed audio and saved transcript to {output_path}"

def op_b9(task: str) -> str:
    """
    B9. Convert Markdown to HTML.
    Expects the task to optionally include:
      - markdown_file: URL or local path of the Markdown file 
                      (default: "https://raw.githubusercontent.com/adam-p/markdown-here/master/README.md")
      - output_html: Output HTML file path (default: "data/output.html")
    
    If the markdown_file is a URL, it is downloaded before conversion.
    Converts the Markdown content to HTML and saves the result.
    """
    system_prompt = (
        "Extract the following parameters from the task as JSON (both parameters are optional): "
        "{'markdown_file': <string, default 'https://raw.githubusercontent.com/adam-p/markdown-here/master/README.md'>, "
        "'output_html': <string, default 'data/output.html'>}."
        "For example: {\"markdown_file\": \"https://raw.githubusercontent.com/adam-p/markdown-here/master/README.md\", "
        "\"output_html\": \"data/output.html\"}."
    )
    params = llm_extract_params(system_prompt, task)
    markdown_file = params.get("markdown_file") or "https://raw.githubusercontent.com/adam-p/markdown-here/master/README.md"
    output_html = params.get("output_html") or "data/output.html"
    
    # Determine if markdown_file is a URL or local file.
    if markdown_file.startswith("http"):
        response = requests.get(markdown_file)
        if response.status_code != 200:
            raise Exception("Failed to download markdown file from URL: " + markdown_file)
        md_text = response.text
    else:
        input_path = os.path.join(os.getcwd(), markdown_file)
        if not os.path.exists(input_path):
            raise Exception(f"Markdown file {input_path} does not exist.")
        with open(input_path, "r", encoding="utf-8") as f:
            md_text = f.read()
    
    # Convert Markdown to HTML.
    html_text = markdown.markdown(md_text)
    
    # Save the HTML output.
    output_path = os.path.join(os.getcwd(), output_html)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    
    return f"Converted Markdown from {markdown_file} to HTML and saved to {output_path}"

def op_b10(task: str) -> str:
    """
    B10. Write an API endpoint that filters a CSV file and returns JSON data.
    Expects the task to optionally include:
      - csv_file: URL or local path to the CSV file 
                  (default: "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv")
      - filter_column: The column name to filter on (default: "Region")
      - filter_value: The value to filter for (default: "Europe")
      - output_json: Output JSON file path (default: "data/output.json")
    
    If csv_file is a URL, it is downloaded before processing.
    Filters the CSV rows where the column equals the specified value,
    then saves the filtered rows as JSON.
    """
    system_prompt = (
        "Extract the following parameters from the task as JSON (all parameters are optional): "
        "{'csv_file': <string, default 'https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv'>, "
        "'filter_column': <string, default 'Region'>, "
        "'filter_value': <string, default 'Europe'>, "
        "'output_json': <string, default 'data/output.json'>}."
        "For example: {\"csv_file\": \"https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv\", "
        "\"filter_column\": \"Region\", \"filter_value\": \"Europe\", \"output_json\": \"data/output.json\"}."
    )
    params = llm_extract_params(system_prompt, task)
    csv_file = params.get("csv_file") or "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
    output_json = params.get("output_json") or "data/output.json"
    
    # Determine if csv_file is a URL or a local file.
    if csv_file.startswith("http"):
        response = requests.get(csv_file)
        if response.status_code != 200:
            raise Exception("Failed to download CSV file from URL: " + csv_file)
        # Save to a temporary file.
        local_csv_path = os.path.join(os.getcwd(), "data", "temp.csv")
        os.makedirs(os.path.dirname(local_csv_path), exist_ok=True)
        with open(local_csv_path, "wb") as f:
            f.write(response.content)
    else:
        local_csv_path = os.path.join(os.getcwd(), csv_file)
        if not os.path.exists(local_csv_path):
            raise Exception(f"CSV file {local_csv_path} does not exist.")
    
    # Load the CSV using pandas.
    df = pd.read_csv(local_csv_path)
    
    # Save the entire DataFrame as JSON.
    output_path = os.path.join(os.getcwd(), output_json)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_json(output_path, orient="records", indent=2)
    
    # Clean up temporary file if CSV was downloaded.
    if csv_file.startswith("http"):
        os.remove(local_csv_path)
    
    return f"Converted CSV from {csv_file} to JSON and saved to {output_path}"

def op_b5(task: str) -> str:
    """
    B5. Run a SQL query on a default SQLite database.
    
    This function expects the task to optionally include:
      - query: The SQL query to run (default: "SELECT name FROM sqlite_master WHERE type='table'")
    
    If the default database (data/default.db) does not exist, a sample SQLite database is created with a sample_table.
    The query is then executed on this database, and the results are saved as JSON to data/query_result.json.
    """
    system_prompt = (
        "Extract the following parameter from the task as JSON (optional): "
        "{'query': <string, default \"SELECT name FROM sqlite_master WHERE type='table'\">}.\n"
        "For example: {\"query\": \"SELECT * FROM sample_table\"}."
    )
    params = llm_extract_params(system_prompt, task)
    query = params.get("query") or "SELECT name FROM sqlite_master WHERE type='table'"

    # Define default database path.
    default_db_path = os.path.join(os.getcwd(), "data", "default.db")
    os.makedirs(os.path.dirname(default_db_path), exist_ok=True)
    
    # If the default DB does not exist, create it with a sample table.
    if not os.path.exists(default_db_path):
        conn = sqlite3.connect(default_db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS sample_table (id INTEGER PRIMARY KEY, name TEXT)")
        cursor.execute("INSERT INTO sample_table (name) VALUES ('Alice'), ('Bob')")
        conn.commit()
        conn.close()
    
    # Execute the query on the default database.
    conn = sqlite3.connect(default_db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
    except Exception as e:
        conn.close()
        raise Exception("Failed to execute query: " + str(e))
    conn.close()
    
    # Save the results as JSON.
    output_path = os.path.join(os.getcwd(), "data", "query_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    return f"Executed SQL query on default SQLite DB and saved results to {output_path}"

def op_convert_json_to_csv(task: str) -> str:
    """
    Convert a JSON file to CSV.
    Expects the task to optionally include:
      - json_url: URL or local path to the JSON file (default: "https://jsonplaceholder.typicode.com/todos")
      - output_csv: Output CSV file path (default: "data/output.csv")
    
    If json_url is a URL, the function downloads it, converts JSON to CSV using pandas, and saves the CSV.
    """
    system_prompt = (
        "Extract the following parameters from the task as JSON (both parameters are optional): "
        "{'json_url': <string, default 'https://jsonplaceholder.typicode.com/todos'>, "
        "'output_csv': <string, default 'data/output.csv'>}."
    )
    params = llm_extract_params(system_prompt, task)
    json_url = params.get("json_url") or "https://jsonplaceholder.typicode.com/todos"
    output_csv = params.get("output_csv") or "data/output.csv"
    
    # Determine if json_url is a URL or local file.
    if json_url.startswith("http"):
        response = requests.get(json_url)
        if response.status_code != 200:
            raise Exception("Failed to download JSON from URL: " + json_url)
        data = response.json()
    else:
        local_json_path = os.path.join(os.getcwd(), json_url)
        if not os.path.exists(local_json_path):
            raise Exception(f"JSON file {local_json_path} does not exist.")
        with open(local_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    
    # Convert JSON data to a DataFrame and then to CSV.
    df = pd.DataFrame(data)
    output_path = os.path.join(os.getcwd(), output_csv)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return f"Converted JSON from {json_url} to CSV and saved to {output_path}"

def op_translate_text(task: str) -> str:
    """
    Translate text to another language using the LLM.
    Expects the task to optionally include:
      - text: The text to translate (default: "Hello, how are you?")
      - target_lang: The target language code (default: "es")
    
    The function sends a prompt to the LLM (via AIRPROXY) asking it to translate the text.
    It returns the translated text as a string.
    """
    # Use LLM to extract parameters from the task
    system_prompt = (
        "Extract the following parameters from the task as JSON (both parameters are optional): "
        "{'text': <string, default 'Hello, how are you?'>, 'target_lang': <string, default 'es'>}."
        "For example: {\"text\": \"Hello, how are you?\", \"target_lang\": \"es\"}."
    )
    params = llm_extract_params(system_prompt, task)
    text = params.get("text") or "Hello, how are you?"
    target_lang = params.get("target_lang") or "es"

    # Build a chat prompt to instruct the LLM to translate the text.
    # We include the target language in the prompt.
    translation_system_prompt = (
        "You are a translation assistant. Translate the following text from English to "
        f"{target_lang}. Return only the translated text as plain text, with no additional commentary."
    )
    # Construct the payload for AIRPROXY's chat completion endpoint
    payload = {
        "model": "gpt-4o-mini",  # Adjust as needed.
        "messages": [
            {"role": "system", "content": translation_system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0
    }
    headers = {
        "Authorization": f"Bearer {os.environ.get('AIPROXY_TOKEN')}",
        "Content-Type": "application/json"
    }
    
    # Send the request to the LLM via AIRPROXY
    chat_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    response = requests.post(chat_url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception("AIRPROXY LLM failed: " + response.text)
    
    result = response.json()
    try:
        assistant_reply = result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise Exception("Failed to parse LLM response: " + str(e))
    
    return f"Translated text: {assistant_reply}"

def op_extract_metadata(task: str) -> str:

    system_prompt = (
        "Extract the following parameter from the task as JSON (optional): "
        "{'image_url': <string, default 'https://i.sstatic.net/l60Hf.png'>}."
    )
    params = llm_extract_params(system_prompt, task)
    image_url = params.get("image_url") or "https://i.sstatic.net/l60Hf.png"
    
    # Download the image.
    response = requests.get(image_url)
    if response.status_code != 200:
        raise Exception("Failed to download image from URL: " + image_url)
    
    try:
        image = Image.open(BytesIO(response.content))
        exif_data = image._getexif()  # May return None if no EXIF data is available.
        if exif_data is None:
            exif_data = {}
    except Exception as e:
        raise Exception("Failed to extract metadata: " + str(e))
    
    return f"Extracted metadata: {json.dumps(exif_data, indent=2)}"

def op_generate_password(task: str) -> str:

    system_prompt = (
        "Extract the following parameters from the task as JSON (both parameters are optional): "
        "{'username': <string, default 'defaultUser'>, 'length': <int, default 16>}."
    )
    params = llm_extract_params(system_prompt, task)
    username = params.get("username") or "defaultUser"
    try:
        length = int(params.get("length") or 16)
    except Exception:
        length = 16

    # Create a pool of characters: uppercase, lowercase, digits, and punctuation.
    pool = string.ascii_letters + string.digits + string.punctuation
    # Generate a random password.
    password = ''.join(random.SystemRandom().choice(pool) for _ in range(length))
    # Optionally, incorporate the username into the password as a prefix/suffix.
    # Here, we'll simply prepend the username (if you wish, you can add more complex logic).
    final_password = f"{username}_{password}"
    
    return f"Generated password: {final_password}"


###############################################
# Task Parsing / Dispatching Function
###############################################

def process_operation_task(task: str) -> str:
    """
    Use AIRPROXY to have an LLM parse the plain‑English task and determine
    which operation (A1–A10) to run.

    The call is made to AIRPROXY using the text-embedding model.
    We expect the LLM to return a JSON with an "operation" field.
    """
    system_prompt = (
        "You are an automation agent that supports the following operations:\n\n"
        "A1: Install a required package and run a remote data generation script with a user email. or NOTE: This will generate data files required for the next tasks.) \n"
        "A2: Format a markdown file in place using a formatter (e.g., Prettier).\n"
        "A3: Process a date file by counting occurrences of specific weekdays and write the count to an output file.\n"
        "A4: Sort an array of contacts in a JSON file by last name then first name and write the sorted output.\n"
        "A5: Read the first line from the 10 most recent log files and write them to an output file.\n"
        "A6: Find all Markdown files in a directory, extract the first H1 from each, and create an index file.\n"
        "A7: Extract the sender’s email address from an email message file and write it to an output file.\n"
        "A8: Extract a credit card number from an image by sending the image (or its Base64) to an LLM and write the number.\n"
        "A9: Using embeddings, find the most similar pair of comments in a file and write them to an output file.\n"
        "A10: Query a SQLite database to calculate total sales for tickets of a specific type and write the result.\n\n"
        "B3: Fetch data from an API and save it.\n"
        "B4: Clone a git repository and make a commit.\n"
        "B5: Run a SQL query on a SQLite or DuckDB database.\n"
        "B6: Extract data from (i.e. scrape) a website.\n"
        "B7: Compress or resize an image.\n"
        "B8: Transcribe audio from an MP3 file.\n"
        "B9: Convert Markdown to HTML.\n"
        "B10: Write an API endpoint that filters a CSV file and returns JSON data.\n"
        "B11: Convert JSON file to CSV .\n"
        "B12: Translate Text to Another Language.\n"
        "B13: Extract Metadata from Files.\n"
        "B14: Dynamic most complext Password generator .\n\n"
        "Based on the user's task description, determine which one of these operations to execute."
        "Return your answer strictly as a JSON object with a single field 'operation' whose value is the operation code "
        "(e.g., {\"operation\": \"A3\"}). Do not include any additional text."
    )
    
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ],
        "temperature":0
    }

    response = requests.post(AIPROXY_URL, headers=headers, json=data)
    print("hello",response)
    if response.status_code != 200:
        raise Exception("AIRPROXY LLM failed: " + response.text)
    
    result = response.json()
    print("hello2",result)
    assistant_content = result["choices"][0]["message"]["content"]
    print(assistant_content)
    parsed_content = json.loads(assistant_content)
    operation = parsed_content.get("operation")

    if not operation:
       raise Exception("AIRPROXY LLM did not return an operation.")
    
    print("operation to run is",operation)

    if operation == "A1":
        print(operation)
        return op_a1(task)
    elif operation == "A2":
        return op_a2(task)
    elif operation == "A3":
        return op_a3(task)
    elif operation == "A4":
        return op_a4(task)
    elif operation == "A5":
        return op_a5(task)
    elif operation == "A6":
        return op_a6(task)
    elif operation == "A7":
        return op_a7(task) 
    elif operation == "A8":
        return op_a8(task) 
    elif operation == "A9":
        return op_a9(task)
    elif operation == "A10":
        return op_a10(task)
    elif operation == "B3":
        return op_b3(task)
    elif operation == "B4":
        return op_b4(task)
    # elif operation == "B5":
    #     return op_b5(task)
    elif operation == "B6":
        return op_b6(task)
    elif operation == "B7":
        return op_b7(task)
    elif operation == "B8":
        return op_b8(task)
    elif operation == "B9":
        return op_b9(task)
    elif operation == "B10":
        return op_b10(task)
    elif operation == "B11":
        return op_convert_json_to_csv(task)
    elif operation == "B12":
        return op_translate_text(task)
    elif operation == "B13":
        return op_extract_metadata(task)
    elif operation == "B14":
        return op_generate_password(task)
    else:
        raise ValueError("Unrecognized operation from LLM: " + str(task))

###############################################
# Flask API Endpoints
###############################################

@app.route('/run', methods=['POST'])
def run_task():
    task = request.args.get('task')
    if not task:
        return make_response("Missing task parameter", 400)
    
    try:
        result = process_operation_task(task)
        # file_hash = hashlib.md5(task.encode()).hexdigest()
        # artifact_path = os.path.join(ARTIFACTS_DIR, f"{file_hash}.txt")
        # with open(artifact_path, "w") as f:
        #     f.write(result)
        
        # response_data = {
        #     "message": "Task executed successfully",
        #     "result": result,
        #     "artifact_path": artifact_path
        # }
        return jsonify(result), 200
    except ValueError as ve:
        return make_response(str(ve), 400)
    except Exception as e:
        return make_response("Internal Server Error: " + str(e), 500)

@app.route('/read', methods=['GET'])
def read_file():
    file_path = request.args.get('path')
    if not file_path:
        return make_response("Missing path parameter", 400)
    
    # Optional: add path sanitization here if needed.

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return make_response("", 404)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/plain'}
    except Exception as e:
        return make_response("Internal Server Error: " + str(e), 500)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)