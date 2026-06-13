from ultralytics import YOLO
import os

# ── CONFIG ──────────────────────────────────────────────
MODEL      = "yolov8n-cls.pt"   # nano = fast; swap to yolov8s-cls.pt for better accuracy
DATA_PATH  = os.path.abspath("data/preprocessed")
EPOCHS     = 50
IMG_SIZE   = 224
BATCH      = 16
PROJECT    = "runs/classify"
RUN_NAME   = "weld_v2_preprocessed"
# ────────────────────────────────────────────────────────

def train():
    model = YOLO(MODEL)

    results = model.train(
        data    = DATA_PATH,
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH,
        project = PROJECT,
        name    = RUN_NAME,
        patience= 10,          # early stopping
        save    = True,
        plots   = True,        # confusion matrix, F1 curve saved automatically
    )

    print("\n✅ Training complete.")
    print(f"Best weights: runs/classify/{RUN_NAME}/weights/best.pt")

if __name__ == "__main__":
    train()