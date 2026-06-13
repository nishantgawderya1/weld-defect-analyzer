import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os

CLASS_NAMES = ["CR", "LP", "PO", "ND"]

class YOLOClassifierWrapper(torch.nn.Module):
    """
    A lightweight wrapper around the YOLOv8 PyTorch model 
    to ensure standard tensor forward pass for Grad-CAM.
    """
    def __init__(self, yolo_model):
        super().__init__()
        self.pytorch_model = yolo_model.model
        
    def forward(self, x):
        res = self.pytorch_model(x)
        if isinstance(res, tuple):
            return res[0]
        return res


def generate_gradcam(model_path: str, image_path: str, output_path: str):
    """
    Generates Grad-CAM heatmap overlay on the input X-ray.
    Shows which regions drove the model's prediction.
    """
    if not os.path.exists(model_path):
        print(f"[WARNING] Model weights not found at: {model_path}. Skipping.")
        return False

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found at: {image_path}")
        return False

    print(f"[INFO] Loading model {model_path}...")
    yolo_model = YOLO(model_path)
    
    # Run standard inference first to get predicted class & confidence
    results = yolo_model(image_path, verbose=False)
    probs   = results[0].probs
    top_cls = probs.top1
    conf    = probs.top1conf.item()

    # Load and resize image to match model input shape (224x224)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    # Convert to torch tensor format
    input_tensor = torch.from_numpy(rgb_img.transpose(2, 0, 1)).unsqueeze(0).float()

    # Wrap the model and extract target layer
    wrapper = YOLOClassifierWrapper(yolo_model)
    
    # Ensure gradients are enabled for model parameters so Grad-CAM can compute gradients
    for param in wrapper.pytorch_model.parameters():
        param.requires_grad = True
        
    wrapper.eval()
    
    # YOLO classification models have self.model.model as nn.Sequential
    # The last feature extraction block is index -2 (just before classification head -1)
    target_layers = [wrapper.pytorch_model.model[-2]]

    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    print(f"[INFO] Generating Grad-CAM heatmap...")
    cam = GradCAM(model=wrapper, target_layers=target_layers)
    
    # Generate heatmap for the top class index
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    # Overlay heatmap onto the RGB image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    output = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

    # Add classification text label overlay
    label = f"{CLASS_NAMES[top_cls]} ({conf*100:.1f}%)"
    cv2.putText(output, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, output)
    print(f"[SUCCESS] Grad-CAM heatmap saved successfully to: {output_path}")
    return True


if __name__ == "__main__":
    # Resolve absolute paths from workspace root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Try different potential model weight locations for weld_v1 (baseline)
    model_locations = [
        os.path.join(base_dir, "runs", "classify", "runs", "classify", "weld_v1", "weights", "best.pt"),
        os.path.join(base_dir, "runs", "classify", "weld_v1", "weights", "best.pt")
    ]
    
    v1_weights = None
    for loc in model_locations:
        if os.path.exists(loc):
            v1_weights = loc
            break
            
    # Sample CR (Crack) image
    sample_cr_image = os.path.join(base_dir, "data", "processed", "test", "CR", "bam5_Img2_A80_S5_[3][10].png")
    
    # Generate for weld_v1 (baseline)
    if v1_weights:
        v1_out = os.path.join(base_dir, "outputs", "gradcam_baseline_cr.png")
        generate_gradcam(
            model_path=v1_weights,
            image_path=sample_cr_image,
            output_path=v1_out
        )
    else:
        print("[WARNING] Could not find baseline weld_v1 weights. Skipping baseline Grad-CAM.")
        
    # Generate for weld_v2_preprocessed
    v2_model_locations = [
        os.path.join(base_dir, "runs", "classify", "runs", "classify", "weld_v2_preprocessed", "weights", "best.pt"),
        os.path.join(base_dir, "runs", "classify", "weld_v2_preprocessed", "weights", "best.pt")
    ]
    
    v2_weights = None
    for loc in v2_model_locations:
        if os.path.exists(loc):
            v2_weights = loc
            break
            
    if v2_weights and os.path.exists(v2_weights):
        v2_out = os.path.join(base_dir, "outputs", "gradcam_preprocessed_cr.png")
        generate_gradcam(
            model_path=v2_weights,
            image_path=sample_cr_image,
            output_path=v2_out
        )
    else:
        print("[INFO] preprocessed weld_v2 weights not found yet (training might still be in progress).")
