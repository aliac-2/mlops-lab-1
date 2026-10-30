import os
import requests
import streamlit as st

INFERENCE_URL = os.environ.get(
    "INFERENCE_URL", "http://127.0.0.1:8000"
)

st.title("Food-11 classifier")

uploaded = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    st.image(uploaded, width=300)

    files = {
        "file": (
            uploaded.name,
            uploaded.getvalue(),
            uploaded.type
        )
    }

    try:
        response = requests.post(
            f"{INFERENCE_URL}/predict",
            files=files,
            timeout=60
        )

        response.raise_for_status()
        result = response.json()

        class_id = result["class_id"]
        confidence = result["confidence"]

        st.success(
            f"Prediction: Class {class_id} "
            f"({confidence:.1%})"
        )

    except requests.RequestException as e:
        st.error(f"Inference service error: {e}")

    except (KeyError, ValueError, TypeError) as e:
        st.error(f"Unexpected prediction response: {e}")