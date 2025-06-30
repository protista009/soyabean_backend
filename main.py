from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import torch
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import io
import platform
import pathlib
import uvicorn 
import os

# Fix for Windows path issues


if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath


app = FastAPI()

# Allow CORS for all origins (you can restrict it to your frontend domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thresholds
LEAF_CONF_THRESH = 0.6
HEALTH_CONF_THRESH = 0.6

# Class labels
HEALTH_LABELS = ['healthy', 'unhealthy']
DISEASE_LABELS = [
    "Healthy", "Mossaic Virus", "Southern blight", "Sudden Death Syndrome",
    "Yellow Mosaic", "bacterial_blight", "brown_spot", "ferrugen",
    "powdery_mildew", "septoria"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct full model paths
HEALTH_MODEL_PATH = os.path.join(BASE_DIR,"mobilenetv2_healthy_best.h5")
DISEASE_MODEL_PATH = os.path.join(BASE_DIR,"mobilenetv2_soybean_best_old.h5")
YOLO_PATH = os.path.join(BASE_DIR,"best.pt")

# Load models
leaf_model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_PATH, force_reload=False)
leaf_model.conf = 0.6

health_model = load_model(HEALTH_MODEL_PATH)
disease_model = load_model(DISEASE_MODEL_PATH)

def read_image(upload_file: UploadFile) -> np.ndarray:
    image = Image.open(io.BytesIO(upload_file.file.read())).convert("RGB")
    return np.array(image)

def preprocess_img(img):
    img = cv2.resize(img, (224, 224))
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return preprocess_input(img)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_array = read_image(file)

        # Step 1: Leaf detection
        results = leaf_model(img_array)
        detections = results.pandas().xyxy[0]
        leaf_detections = detections[detections['name'] == 'leaf']

        if leaf_detections.empty:
            raise HTTPException(status_code=400, detail="No leaf detected with sufficient confidence.")

        best_leaf = leaf_detections.iloc[0]
        x1, y1, x2, y2 = map(int, [best_leaf['xmin'], best_leaf['ymin'], best_leaf['xmax'], best_leaf['ymax']])
        leaf_crop = img_array[y1:y2, x1:x2]

        # Step 2: Health classification
        health_input = preprocess_img(leaf_crop)
        health_pred = health_model.predict(np.expand_dims(health_input, axis=0), verbose=0)[0]
        health_idx = np.argmax(health_pred)
        health_status = HEALTH_LABELS[health_idx]
        health_conf = float(health_pred[health_idx])

        if health_status == "healthy" or health_conf < HEALTH_CONF_THRESH:
            return {
                "result": health_status,
                "confidence": health_conf,
                "stage": "health"
            }

        # Step 3: Disease classification
        disease_input = preprocess_img(leaf_crop)
        disease_pred = disease_model.predict(np.expand_dims(disease_input, axis=0), verbose=0)[0]
        disease_idx = np.argmax(disease_pred)
        disease = DISEASE_LABELS[disease_idx]
        disease_conf = float(disease_pred[disease_idx])

        return {
            "result": disease,
            "confidence": disease_conf,
            "stage": "disease"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Run with: uvicorn main:app --reload
