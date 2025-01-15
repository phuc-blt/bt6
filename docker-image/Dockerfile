# Chọn image cơ sở từ Python 3.9
FROM python:3.9-slim

# Cài đặt các thư viện hệ thống yêu cầu
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép requirements.txt vào trong container
COPY requirements.txt /app/

# Cài đặt các thư viện phụ thuộc
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn vào container
COPY . /app/

# Cổng mà FastAPI sẽ chạy trên
EXPOSE 5000

# Lệnh khởi chạy ứng dụng FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
