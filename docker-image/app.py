from fastapi import FastAPI, File, UploadFile
import uvicorn
import torch
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io
from ultralytics import YOLO
from PIL import Image

app = FastAPI()

model = YOLO("last.pt")  # Replace "best.pt" with the correct path to your trained model


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load and preprocess the image
    image = Image.open(io.BytesIO(await file.read()))
    image = image.resize((640, 640))  # Resize to model's input size
    input_data = np.array(image).astype(np.float32) / 255.0  # Normalize
    input_data = np.transpose(input_data, (2, 0, 1))  # HWC to CHW
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(input_data).float()

    # Perform inference
    with torch.no_grad():
        results = model(input_tensor)

    # Process the model output (assuming the model returns bounding boxes and scores)
    output = results.xyxy[0]  # Bounding boxes in [x1, y1, x2, y2, conf, class]

    # Extract data
    boxes = output[:, :4]  # Bounding box coordinates
    scores = output[:, 4]  # Confidence scores
    labels = output[:, 5]  # Class labels

    # Filter predictions by confidence threshold (e.g., 0.5)
    confidence_threshold = 0.01
    valid_indices = torch.where(scores > confidence_threshold)[0]

    predictions = []
    for idx in valid_indices:
        box = boxes[idx].tolist()  # Get the bounding box for this prediction
        score = scores[idx].item()  # Confidence score
        label = labels[idx].item()  # Class label
        predictions.append({
            "box": box,
            "score": float(score),
            "label": int(label)
        })

    return {"predictions": predictions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
