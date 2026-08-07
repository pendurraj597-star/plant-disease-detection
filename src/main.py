# ============================================================
# Plant Disease Detection - Main Program
# ============================================================

from src.predict import predict_disease


def main():
    """
    Ask the user for an image and predict
    the plant disease.
    """

    image_path = input(
        "Enter the path of the plant image: "
    ).strip()

    try:

        disease, confidence = predict_disease(
            image_path
        )

        print()
        print("==============================")
        print("   PLANT DISEASE RESULT")
        print("==============================")

        print(
            f"Disease/Class: {disease}"
        )

        print(
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )

        print("==============================")

    except Exception as error:

        print()
        print(
            f"Error: {error}"
        )


if __name__ == "__main__":
    main()
