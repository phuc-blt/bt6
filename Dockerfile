FROM apache/airflow:2.10.3

# Switch to root to install system dependencies
USER root
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx

# Switch back to airflow user
USER airflow

# Copy and install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
