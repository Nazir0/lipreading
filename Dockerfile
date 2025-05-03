# Use an official Python runtime as a parent image
# Using slim reduces image size, Python 3.11 as an example
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5001

# Install system dependencies needed by OpenCV and potentially MediaPipe
# Use non-interactive frontend to avoid prompts during build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    # Add any other system dependencies identified during testing
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# This includes app.py, templates/, static/, model_files/
# (Ensure static/videos and model_files are NOT in .dockerignore if you want them inside)
COPY . .

# Make port 5001 available to the world outside this container
EXPOSE 5001

# Define the command to run your app using Flask's built-in server
# Use gunicorn or waitress for production
CMD ["flask", "run"]
# Or directly using python: CMD ["python", "app.py"]
