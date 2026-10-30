
import os
import io

import mlflow
import numpy as np

from PIL import Image
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms


app = FastAPI()


# MLflow server address
mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )
)


# Load the registered model using the champion alias
model = mlflow.pyfunc.load_model(
    "models:/food11@champion"
)


# ResNet18 preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read image
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # Apply preprocessing
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Convert to numpy for MLflow pyfunc
    image_array = image_tensor.numpy()

    # Get model output
    prediction = model.predict(
        image_array
    )

    # Take first sample
    scores = np.asarray(prediction[0], dtype=np.float64)

    # Get predicted class
    class_id = int(np.argmax(scores))

    # Convert raw logits to probabilities using softmax
    exp_scores = np.exp(scores - np.max(scores))
    probabilities = exp_scores / np.sum(exp_scores)

    confidence = float(probabilities[class_id])

    return {
        "class_id": class_id,
        "confidence": confidence
    }
