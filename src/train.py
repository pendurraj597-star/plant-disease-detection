# ============================================================
# Plant Disease Detection - Model Training
# ============================================================

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from src.utils import (
    IMAGE_SIZE,
    create_directories
)


# Dataset directories.
TRAIN_DIR = Path("dataset/train")
VALIDATION_DIR = Path("dataset/validation")

# Saved model location.
MODEL_PATH = Path(
    "model/plant_disease_model.keras"
)

# Training configuration.
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42


def build_model(number_of_classes):
    """
    Build a CNN model for plant disease classification.
    """

    model = models.Sequential([

        # Input image.
        layers.Input(
            shape=(
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
                3
            )
        ),

        # Normalize image pixels.
        layers.Rescaling(1.0 / 255),

        # First convolution block.
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),
        layers.MaxPooling2D(),

        # Second convolution block.
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),
        layers.MaxPooling2D(),

        # Third convolution block.
        layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),
        layers.MaxPooling2D(),

        # Convert feature maps into one-dimensional data.
        layers.Flatten(),

        # Reduce overfitting.
        layers.Dropout(0.4),

        # Fully connected layer.
        layers.Dense(
            128,
            activation="relu"
        ),

        # Output layer.
        # Softmax gives probability for each disease class.
        layers.Dense(
            number_of_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    """
    Train the plant disease classification model.
    """

    create_directories()

    # Check whether training data exists.
    class_folders = [
        folder for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
    ]

    if not class_folders:
        raise FileNotFoundError(
            "Please add plant images inside "
            "dataset/train/<disease-name>/"
        )

    print("Loading training dataset...")

    # Load training images.
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    # Load validation images.
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    # Save class names for prediction.
    class_names = train_dataset.class_names

    print("Detected classes:")
    print(class_names)

    # Improve data loading.
    autotune = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(
        autotune
    )

    validation_dataset = validation_dataset.prefetch(
        autotune
    )

    # Build CNN.
    model = build_model(
        len(class_names)
    )

    model.summary()

    # Stop training when validation loss stops improving.
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )

    # Save best model.
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True
    )

    print("Starting training...")

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[
            early_stopping,
            checkpoint
        ]
    )

    # Save class names along with the model.
    with open(
        "model/class_names.txt",
        "w",
        encoding="utf-8"
    ) as file:

        for class_name in class_names:
            file.write(
                class_name + "\n"
            )

    print()
    print("Training completed!")
    print(
        f"Model saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
