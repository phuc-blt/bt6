from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import subprocess
import logging
from bing_image_downloader import downloader
import shutil
import json
import pandas as pd
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
def crawl_tomato_images(output_folder="data_lake", num_images=50):
    """Crawl tomato images using Bing Image Downloader."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    search_term = "tomato"
    logging.info(f"Downloading {num_images} images from Bing with keyword: {search_term}...")
    downloader.download(
        search_term,
        limit=num_images,
        output_dir=output_folder,
        adult_filter_off=True,
        force_replace=False,
        timeout=60
    )
    logging.info(f"Images have been saved to: {output_folder}/{search_term}")

def predict_and_label(data_lake, data_pool):
    """Predict and label images in the data lake folder."""
    if not os.path.exists(data_pool):
        os.makedirs(data_pool)

    for image in os.listdir(data_lake):
        if image.endswith(('.jpg', '.png')):  # Process only image files
            image_path = os.path.join(data_lake, image)

            # Simulated API response
            response = {
                "image": image,
                "label": "tomato",
                "confidence": 0.95
            }

            # JSON label file path
            label_file = os.path.join(data_pool, image.rsplit(".", 1)[0] + ".json")

            if os.path.exists(label_file):
                logging.info(f"Label already exists for {image}, skipping.")
                continue

            with open(label_file, "w") as f:
                json.dump(response, f)

            logging.info(f"Processed and labeled: {image}")

def eval_model(current_day, EXPERIMENT_DIR, WEIGHT_DIR):
    """Evaluate and update model if the new one performs better."""
    logging.info("=== Starting model evaluation ===")

    current_result_path = os.path.join(EXPERIMENT_DIR, f'{current_day}', 'results.csv')
    best_result_path = os.path.join(WEIGHT_DIR, 'results.csv')

    def getMAP(info):
        logging.info(f"Reading data from file: {info}")
        df = pd.read_csv(info)
        logging.info(f"File content:\n{df.tail()}")
        last_epoch = df.tail(1)
        mAP50 = last_epoch['metrics/mAP50(B)']
        return mAP50

    try:
        cur_mAP50 = float(getMAP(current_result_path).iloc[0])
        pre_mAP50 = float(getMAP(best_result_path).iloc[0])
        logging.info(f"Current mAP50: {cur_mAP50}, Previous mAP50: {pre_mAP50}")
    except Exception as e:
        logging.error(f"Error reading files or calculating mAP50: {e}")
        return

    if cur_mAP50 >= pre_mAP50:
        logging.info("New model performs better or equally. Updating weights.")
        try:
            pre_weight_path = os.path.join(f'{WEIGHT_DIR}', 'best.pt')

            if os.path.exists(pre_weight_path):
                os.remove(pre_weight_path)
                logging.info(f"Deleted old weight file: {pre_weight_path}")
            if os.path.exists(best_result_path):
                os.remove(best_result_path)
                logging.info(f"Deleted old result file: {best_result_path}")

            shutil.copy(current_result_path, WEIGHT_DIR)
            cur_weight_path = os.path.join(EXPERIMENT_DIR, f'{current_day}', 'weights', 'best.pt')
            shutil.copy(cur_weight_path, WEIGHT_DIR)
            logging.info(f"Copied new weights from: {cur_weight_path} to {WEIGHT_DIR}")
            logging.info(f"Copied new results from: {current_result_path} to {WEIGHT_DIR}")
            logging.info("=== Update completed ===")
        except Exception as e:
            logging.error(f"Error during file copy or deletion: {e}")
    else:
        logging.info("New model does not perform better. No changes made.")

def train_model(data_lake, EXPERIMENT_DIR, WEIGHT_DIR):
    """Train the YOLO model."""
    logging.info("=== Starting model training ===")

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
            label = 0  # Assuming label is tomato (0)

            if self.transform:
                image = self.transform(image)

            return image, label

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = TomatoDataset(data_lake, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = YourModel()  # Replace with your YOLOv11 model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    logging.info("=== Training completed ===")

    current_day = datetime.now().strftime("%Y_%m_%d")
    model_save_path = os.path.join(EXPERIMENT_DIR, current_day, "weights", "best.pt")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Saved trained model at {model_save_path}")

    eval_model(current_day, EXPERIMENT_DIR, WEIGHT_DIR)

# DAG definitions
with DAG(
    "crawl_tomato_images",
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)},
    description="Crawl tomato images every day at 9 AM",
    schedule_interval="0 9 * * *",
    start_date=datetime(2025, 1, 13),
    catchup=False,
) as dag_crawl:

    crawl_task = PythonOperator(
        task_id="crawl_tomato_images",
        python_callable=crawl_tomato_images,
        op_kwargs={"output_folder": "data_lake", "num_images": 50},
    )

with DAG(
    "predict_tomato_images",
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)},
    description="Predict tomato images every hour if sufficient data",
    schedule_interval="@hourly",
    start_date=datetime(2025, 1, 13),
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
    start_date=datetime(2025, 1, 13),
    catchup=False
) as dag_train:

    def check_and_train():
        data_pool = "/opt/airflow/data_pool"
        if len(os.listdir(data_pool)) >= 50:
            train_model(data_pool, EXPERIMENT_DIR, WEIGHT_DIR)

    train_and_update_task = PythonOperator(
        task_id="train_and_update_model",
        python_callable=check_and_train,
    )
