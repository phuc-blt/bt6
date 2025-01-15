from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from bing_image_downloader import downloader
import os
import requests
import json
import subprocess
import pandas as pd
import shutil
import logging

# Default arguments for DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Function: Crawl tomato images
def crawl_tomato_images(output_folder="data_lake", num_images=100):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    search_term = "tomato"
    logging.info(f"Đang tải {num_images} ảnh từ Bing với từ khóa: {search_term}...")
    downloader.download(
        search_term,
        limit=num_images,
        output_dir=output_folder,
        adult_filter_off=True,
        force_replace=False,
        timeout=60
    )
    logging.info(f"Ảnh đã được lưu trong thư mục: {output_folder}/{search_term}")

# Function: Process images from data_lake
def process_images_from_datalake():
    image_folder = "/opt/airflow/data_lake/tomato"
    output_folder = "/opt/airflow/data_pool"
    api_url = "http://0.0.0.0:5000/predict"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    if len(images) >= 50:
        for img in images[:50]:
            with open(os.path.join(image_folder, img), 'rb') as file:
                response = requests.post(api_url, files={'image': file})
                labels = response.json()
                with open(os.path.join(output_folder, f"{os.path.splitext(img)[0]}.txt"), 'w') as label_file:
                    json.dump(labels, label_file)

# Function: Train a new model
def train_new_model():
    data_pool = "/opt/airflow/data_pool"
    model_dir = "/opt/airflow/model"
    weight_path = os.path.join(model_dir, "best.pt")  # Đường dẫn tới best.pt
    experiment_dir = "experiments"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    labels = [f for f in os.listdir(data_pool) if f.endswith('.json')]
    if len(labels) >= 50:
        logging.info("Bắt đầu huấn luyện mô hình mới...")
        
        # Cấu hình dữ liệu và các thông số huấn luyện
        data_yaml_path = "data/tomato.yaml"  # File cấu hình dữ liệu YOLO
        output_path = os.path.join(experiment_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(output_path, exist_ok=True)
        
        # Lệnh huấn luyện YOLO
        train_command = [
            "yolo", "train", 
            "--data", data_yaml_path,
            "--weights", weight_path,
            "--epochs", "50",
            "--img-size", "640",
            "--project", output_path,
            "--name", "tomato_experiment"
        ]

        try:
            subprocess.run(train_command, check=True)
            logging.info(f"Mô hình mới được lưu tại: {output_path}")
            
            # So sánh độ chính xác (mAP50)
            current_day = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            eval_model(current_day, experiment_dir, model_dir)

        except subprocess.CalledProcessError as e:
            logging.error(f"Lỗi khi huấn luyện mô hình: {e}")

# Function: Evaluate the model
def eval_model(current_day, EXPERIMENT_DIR, WEIGHT_DIR):
    logging.info("=== Bắt đầu quá trình kiểm tra model ===")

    current_result_path = os.path.join(EXPERIMENT_DIR, f'{current_day}', 'results.csv')
    best_result_path = os.path.join(WEIGHT_DIR, 'results.csv')

    def getMAP(info):
        logging.info(f"Đang đọc thông tin từ file: {info}")
        df = pd.read_csv(info)
        logging.info(f"Nội dung file:\n{df.tail()}")
        last_epoch = df.tail(1)
        mAP50 = last_epoch['metrics/mAP50(B)']
        return mAP50

    try:
        cur_mAP50 = float(getMAP(current_result_path).iloc[0])
        pre_mAP50 = float(getMAP(best_result_path).iloc[0])
        logging.info(f"mAP50 hiện tại: {cur_mAP50}, mAP50 trước đó: {pre_mAP50}")
    except Exception as e:
        logging.error(f"Lỗi khi đọc file hoặc tính toán mAP50: {e}")
        return

    if cur_mAP50 >= pre_mAP50:
        logging.info("Kết quả mới tốt hơn hoặc bằng kết quả cũ. Tiến hành cập nhật trọng số.")
        try:
            pre_weight_path = os.path.join(f'{WEIGHT_DIR}', 'best.pt')

            if os.path.exists(pre_weight_path):
                os.remove(pre_weight_path)
                logging.info(f"Đã xóa file trọng số cũ: {pre_weight_path}")
            if os.path.exists(best_result_path):
                os.remove(best_result_path)
                logging.info(f"Đã xóa file kết quả cũ: {best_result_path}")

            shutil.copy(current_result_path, WEIGHT_DIR)
            cur_weight_path = os.path.join(EXPERIMENT_DIR, f'{current_day}', 'weights', 'best.pt')
            shutil.copy(cur_weight_path, WEIGHT_DIR)
            logging.info(f"Đã sao chép file trọng số mới từ: {cur_weight_path} đến {WEIGHT_DIR}")
            logging.info(f"Đã sao chép file kết quả mới từ: {current_result_path} đến {WEIGHT_DIR}")
            logging.info("=== Quá trình cập nhật hoàn tất ===")
        except Exception as e:
            logging.error(f"Lỗi trong quá trình sao chép hoặc xóa file: {e}")
    else:
        logging.info("Kết quả mới không tốt hơn. Không có thay đổi nào được thực hiện.")

# Define DAG
with DAG(
    dag_id='tomato_pipeline',
    default_args=default_args,
    description='A pipeline to crawl, process, and train models for tomato detection',
    schedule_interval='@hourly',
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    # Task 1: Crawl images
    task_crawl_images = PythonOperator(
        task_id='download_images',
        python_callable=crawl_tomato_images
    )

    # Task 2: Process images
    task_process_images = PythonOperator(
        task_id='process_images',
        python_callable=process_images_from_datalake,
    )
        # Task 3: Train new model
    task_train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_new_model,
    )
    
    task_danhgia = PythonOperator(
        task_id='eval_model',
        python_callable=eval_model,
    )
    # Define task dependencies
    task_crawl_images >> task_process_images >> task_train_model >> task_danhgia
