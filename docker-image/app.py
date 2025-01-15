from fastapi import FastAPI, File, UploadFile
import torch
from pydantic import BaseModel
from typing import List
import io
from ultralytics import YOLO
from PIL import Image

app = FastAPI()

# Load YOLO model
model = YOLO("best.pt") 

class Detection(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    class_label: str

@app.post("/predict", response_model=List[Detection])
async def predict(image: UploadFile = File(...)):
    # Read image file
    content = await image.read()
    img = Image.open(io.BytesIO(content))

    # Inference with YOLO model
    results = model(img)
    detections = results.pandas().xyxy[0]

    # Convert detections to response format
    response = []
    for _, row in detections.iterrows():
        response.append({
            "xmin": row["xmin"],
            "ymin": row["ymin"],
            "xmax": row["xmax"],
            "ymax": row["ymax"],
            "confidence": row["confidence"],
            "class_label": row["name"]
        })

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
