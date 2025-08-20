# Use an official Python runtime
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl gnupg git

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set work directory
WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose port for Flask
EXPOSE 5000

# Start Ollama in background and then Flask
CMD ollama serve & python app.py
