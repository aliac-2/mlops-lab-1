
import io
import os
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

import mlflow

app = FastAPI()

# Configure MLflow server address.
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)

# Load model at startup, not during module import.
model = None


@app.on_event("startup")
def load_model():
    global model

    # Skip real model loading during CI and unit tests.
    if os.environ.get("SKIP_MODEL"):
        model = None
        return

    model = mlflow.pyfunc.load_model("models:/food11@champion")


# ResNet18 preprocessing.
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    # Read uploaded image.
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ResNet18 preprocessing and batch dimension.
    image_tensor = transform(image).unsqueeze(0)
    image_array = image_tensor.numpy()

    # Get model output.
    prediction = model.predict(image_array)
    scores = np.asarray(prediction[0], dtype=np.float64)

    # Get predicted class.
    class_id = int(np.argmax(scores))

    # Convert raw logits to probabilities.
    exp_scores = np.exp(scores - np.max(scores))
    probabilities = exp_scores / np.sum(exp_scores)
    confidence = float(probabilities[class_id])

    return {
        "class_id": class_id,
        "confidence": confidence,
    }
