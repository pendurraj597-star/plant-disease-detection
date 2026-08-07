# ============================================================
# Plant Disease Detection - Prediction
# ============================================================

from pathlib import Path

import numpy as np
import tensorflow as tf

from src.utils import preprocess_image


# Trained model location.
MODEL_PATH = Path(
    "model/plant_disease_model.keras"
)

# Class names file.
CLASS_NAMES_PATH = Path(
    "model/class_names.txt"
)


def load_class_names():
    """
    Load disease names saved during training.
    """

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "Class names file was not found."
        )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        classes = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return classes


def predict_disease(image_path):
    """
    Predict the disease class for a plant image.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. "
            "Run the training program first."
        )

    # Load trained model.
    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # Load disease class names.
    class_names = load_class_names()

    # Prepare image.
    image = preprocess_image(
        image_path
    )

    # Make prediction.
    predictions = model.predict(
        image,
        verbose=0
    )[0]

    # Find class with highest probability.
    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    return predicted_class, confidence
