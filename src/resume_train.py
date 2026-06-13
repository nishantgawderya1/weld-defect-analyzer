from ultralytics import YOLO
import os

def resume_training():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(
        base_dir, 
        "runs", 
        "classify", 
        "runs", 
        "classify", 
        "weld_v2_preprocessed", 
        "weights", 
        "last.pt"
    )
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    print(f"[INFO] Resuming preprocessed training from checkpoint: {checkpoint_path}")
    
    # Load model
    model = YOLO(checkpoint_path)
    
    # Resume training and override epochs to 12
    results = model.train(resume=True, epochs=12)
    print("\n[SUCCESS] Resumed training complete.")
    print(f"Best weights: runs/classify/runs/classify/weld_v2_preprocessed/weights/best.pt")

if __name__ == "__main__":
    resume_training()
