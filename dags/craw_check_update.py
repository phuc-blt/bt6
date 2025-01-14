import os
import logging
import shutil
import json
from datetime import datetime, timedelta
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from bing_image_downloader import downloader
import torch
from train import train_yolo 
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variables
EXPERIMENT_DIR = "/opt/airflow/experiment_dir"
WEIGHT_DIR = "/opt/airflow/weight_dir"

# Helper functions
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


def predict_and_label(data_lake, data_pool):
    # Tạo folder data_pool nếu chưa tồn tại
    if not os.path.exists(data_pool):
        os.makedirs(data_pool)

    for image in os.listdir(data_lake):
        if image.endswith(('.jpg', '.png')):  # Chỉ xử lý file ảnh
            image_path = os.path.join(data_lake, image)

            # Giả lập kết quả API
            response = {
                "image": image,
                "label": "tomato",
                "confidence": 0.95
            }

            # Xử lý tên file JSON
            label_file = os.path.join(data_pool, image.rsplit(".", 1)[0] + ".json")

            # Kiểm tra nếu file nhãn đã tồn tại
            if os.path.exists(label_file):
                print(f"Label already exists for {image}, skipping.")
                continue

            # Lưu nhãn dưới dạng JSON
            with open(label_file, "w") as f:
                json.dump(response, f)

            print(f"Processed and labeled: {image}")

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

def train_model(data_lake, EXPERIMENT_DIR, WEIGHT_DIR):
    logging.info("=== Bắt đầu quá trình huấn luyện mô hình ===")

    # Chuẩn bị dữ liệu (dưới dạng Dataset và DataLoader)
    class TomatoDataset(Dataset):
        def __init__(self, image_folder, transform=None):
            self.image_folder = image_folder
            self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
            self.transform = transform

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_name = os.path.join(self.image_folder, self.image_files[idx])
            image = Image.open(img_name)
            label = 0  # Giả sử label là cà chua (0 - tomato)

            if self.transform:
                image = self.transform(image)

            return image, label

    # Chuyển ảnh thành Tensor và chuẩn hóa
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Thay đổi kích thước ảnh nếu cần
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa theo ImageNet
    ])

    dataset = TomatoDataset(data_lake, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Khởi tạo mô hình và optimizer
    model = YourModel()  # Thay YourModel bằng mô hình của bạn
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Đưa mô hình vào chế độ huấn luyện
    model.train()

    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            # Di chuyển dữ liệu vào GPU nếu có
            inputs, labels = inputs.cuda(), labels.cuda()

            # Đặt gradient về 0
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass và cập nhật trọng số
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    logging.info("=== Quá trình huấn luyện kết thúc ===")

    # Lưu mô hình sau khi huấn luyện
    current_day = datetime.now().strftime("%Y_%m_%d")
    model_save_path = os.path.join(EXPERIMENT_DIR, current_day, "weights", "best.pt")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Đã lưu mô hình huấn luyện tại {model_save_path}")

    # Gọi eval_model để so sánh và cập nhật trọng số tốt nhất nếu cần
    eval_model(current_day, EXPERIMENT_DIR, WEIGHT_DIR)

# DAG definitions
with DAG(
    "crawl_tomato_images",
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)},
    description="Crawl tomato images every day at 9 AM",
    schedule_interval="0 9 * * *",
    start_date=days_ago(1),
    catchup=False,
) as dag_crawl:

    crawl_task = PythonOperator(
        task_id="crawl_tomato_images",
        python_callable=crawl_tomato_images,
        op_kwargs={"output_folder": "data_lake", "num_images": 100},
    )

with DAG(
    "predict_tomato_images",
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)},
    description="Predict tomato images every hour if sufficient data",
    schedule_interval="@hourly",
    start_date=days_ago(1),
    catchup=False,
) as dag_predict:

    def check_and_predict():
        if len(os.listdir("data_lake")) >= 50:
            if not os.path.exists("data_pool"):
                os.makedirs("data_pool")
            predict_and_label("data_lake", "data_pool")

    process_and_label_task = PythonOperator(
        task_id='predict_and_label',
        python_callable=predict_and_label,
        op_args=["/opt/airflow/data_lake/tomato", "/opt/airflow/data_pool"]
    )

with DAG(
    "train_and_update_model",
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)},
    description="Train and update model every hour if sufficient data",
    schedule_interval="@hourly",
    start_date=days_ago(1),
    catchup=False,
) as dag_train:

    def check_and_train(EXPERIMENT_DIR, WEIGHT_DIR):
        current_day = datetime.now().strftime("%Y_%m_%d")
        train_model("/opt/airflow/data_lake/tomato", EXPERIMENT_DIR, WEIGHT_DIR)
        eval_model(current_day, EXPERIMENT_DIR, WEIGHT_DIR)

    train_task = PythonOperator(
        task_id="train_and_update_model",
        python_callable=check_and_train,
        op_kwargs={
            "EXPERIMENT_DIR": EXPERIMENT_DIR,
            "WEIGHT_DIR": WEIGHT_DIR,
        },
    )
