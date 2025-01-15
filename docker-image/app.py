import os
import cv2
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from typing import List
from pathlib import Path
import logging

app = FastAPI()

model = YOLO('last.pt')

class BatchPredictPayload(BaseModel):
    data_folder: str
    output_folder: str

@app.post("/predict/")
async def predict(payload: BatchPredictPayload):
    """
    API to perform batch predictions using YOLO model.
    """
    logging.basicConfig(level=logging.INFO)

    data_folder = payload.data_folder
    output_folder = payload.output_folder

    try:
        if not os.path.exists(data_folder):
            raise HTTPException(status_code=400, detail=f"Data folder '{data_folder}' does not exist.")
        
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        image_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(image_files) < 50:
            raise HTTPException(status_code=400, detail="Not enough images in the data folder. At least 50 images are required.")

        results_summary = []

        for image_file in image_files:
            file_path = os.path.join(data_folder, image_file)

            image = cv2.imread(file_path)
            if image is None:
                logging.warning(f"Cannot read image file: {file_path}. Skipping.")
                continue

            results = model(image)

            predictions = []
            for box in results[0].boxes:  
                bbox = box.xyxy[0].tolist()  
                confidence = box.conf[0].item()  
                label = model.names[int(box.cls[0])]  

                predictions.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": bbox
                })

            # Writing results to a .txt file
            output_file = os.path.join(output_folder, f"{Path(image_file).stem}_predictions.txt")
            with open(output_file, 'w') as txt_file:
                for prediction in predictions:
                    txt_file.write(f"Label: {prediction['label']}\n")
                    txt_file.write(f"Confidence: {prediction['confidence']}\n")
                    txt_file.write(f"Bounding Box: {prediction['bbox']}\n\n")

            results_summary.append({
                "file": image_file,
                "predictions": predictions
            })

        return {"message": "Batch prediction completed successfully.", "results": results_summary}

    except Exception as e:
        logging.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
