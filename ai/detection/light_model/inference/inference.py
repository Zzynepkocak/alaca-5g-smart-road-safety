from ultralytics import YOLO


# Path to trained weights — update after training
# e.g. "outputs/light_model_v1/weights/best.pt"
MODEL_PATH = "outputs/light_model_v1/weights/best.pt"

# Confidence threshold for detections
CONFIDENCE = 0.4

# Device: "cpu", "cuda", or "mps"
DEVICE = "cpu"


def run_inference(source: str):
    """
    Run inference on a given source.

    Args:
        source: Path to image, video, directory, or webcam index (e.g. "0")
    """
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=source,
        conf=CONFIDENCE,
        device=DEVICE,
        save=True,
        project="outputs",
        name="inference_results",
    )

    for result in results:
        print(result.boxes)

    return results


if __name__ == "__main__":
    # Replace with your image/video path or webcam index
    run_inference(source="path/to/image_or_video")
