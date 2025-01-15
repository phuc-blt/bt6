FROM python:3.9

# Set working directory
WORKDIR /app

# Cài đặt thư viện OpenGL (libGL.so.1) để tránh lỗi với OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt thư viện libjpeg-dev và libpng-dev cho PIL (Pillow)
RUN apt-get install -y \
    libjpeg-dev \
    libpng-dev

# Cài đặt các gói cần thiết cho Gradio và FastAPI
RUN apt-get install -y \
    curl \
    && curl -sL https://deb.nodesource.com/setup_14.x | bash - \
    && apt-get install -y nodejs

# Copy các file vào thư mục làm việc
COPY . /app

# Cài đặt các phụ thuộc từ requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Chạy ứng dụng FastAPI và Gradio
CMD ["python", "app.py"]
