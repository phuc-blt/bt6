from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
import shutil
import os

# Load model
MODEL_PATH = "/home/phuc/airflow/model/current_model.pt"
model = YOLO(MODEL_PATH)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Tomato Detection API is running."}

@app.post("/predict/")
async def predict_image(file: UploadFile):
    """
    Predict objects in an uploaded image using the YOLOv11 model.
    """
    # Save the uploaded image
    image_path = "/home/phuc/airflow/outputdata"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run prediction
    results = model(image_path)

    # Parse results
    predictions = []
    for result in results:
        for box in result.boxes:
            predictions.append({
                "label": box.label,
                "confidence": float(box.conf),
                "bbox": box.xywh.tolist()
            })

    # Remove the temporary image
    os.remove(image_path)
    
    return {"predictions": predictions}
