# ============================================================
# Plant Disease Detection - Utility Functions
# ============================================================

from pathlib import Path
import numpy as np
from PIL import Image


# Image size expected by the neural network.
IMAGE_SIZE = (128, 128)


def preprocess_image(image_path):
    """
    Load an image and prepare it for model prediction.
    """

    # Open the image.
    image = Image.open(image_path).convert("RGB")

    # Resize image to the required size.
    image = image.resize(IMAGE_SIZE)

    # Convert image to NumPy array.
    image = np.array(image, dtype=np.float32)

    # Normalize pixel values from 0-255 to 0-1.
    image = image / 255.0

    # Add batch dimension.
    image = np.expand_dims(image, axis=0)

    return image


def create_directories():
    """
    Create project directories if they don't exist.
    """

    Path("dataset/train").mkdir(
        parents=True,
        exist_ok=True
    )

    Path("dataset/validation").mkdir(
        parents=True,
        exist_ok=True
    )

    Path("model").mkdir(
        parents=True,
        exist_ok=True
    )
