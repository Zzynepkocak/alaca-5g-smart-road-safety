from ultralytics import YOLO


# Path to your dataset config (YOLO format)
DATASET_CONFIG = "configs/dataset.yaml"

# Pretrained YOLOv8 nano weights — lightweight baseline
MODEL_WEIGHTS = "yolov8n.pt"

# Training hyperparameters
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16

# Output directory for runs/weights
PROJECT_DIR = "outputs"
RUN_NAME = "light_model_v1"


def train():
    model = YOLO(MODEL_WEIGHTS)

    model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
    )

    print(f"Training complete. Results saved to {PROJECT_DIR}/{RUN_NAME}")


if __name__ == "__main__":
    train()
