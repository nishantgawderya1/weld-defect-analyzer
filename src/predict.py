from ultralytics import YOLO
import os
import torch

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

class WeldPredictor:
    def __init__(self, model_path=None, apply_preprocessing=False):
        """
        Initializes the Weld Defect Predictor.
        Default path points to the best weights trained.

        apply_preprocessing: if True, run the CLAHE + denoise pipeline on every
        input before inference. Required for the weld_v2 model, which was trained
        on preprocessed images — feeding it raw images causes train/serve skew.
        Leave False for weld_v1 (trained on raw images).
        """
        self.apply_preprocessing = apply_preprocessing
        if model_path is None:
            # Resolve default path to best.pt
            model_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "runs", 
                "classify", 
                "runs", 
                "classify", 
                "weld_v1", 
                "weights", 
                "best.pt"
            ))
        
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model weights not found at: {model_path}. Please check training status.")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Class detailed mappings
        self.class_details = {
            "CR": {
                "name": "Crack (CR)",
                "level": "CRITICAL",
                "color": "#EF4444",  # Crimson Red
                "bg_color": "rgba(239, 68, 68, 0.1)",
                "border_color": "#EF4444",
                "badge": "🔴 Critical Failure",
                "desc": "Linear rupture or fracture occurring in the weld metal or heat-affected zone due to high stress, rapid cooling, or chemical impurities.",
                "action": "CRITICAL WELD FAILURE: Crack detected. Cracks propagate rapidly under dynamic loading and lead to catastrophic structural failure. Reject the weld immediately. Perform structural excavation (grinding), stress-relief heat treatment, and completely re-weld."
            },
            "LP": {
                "name": "Lack of Penetration (LP)",
                "level": "SERIOUS",
                "color": "#F59E0B",  # Amber/Orange
                "bg_color": "rgba(245, 158, 11, 0.1)",
                "border_color": "#F59E0B",
                "badge": "🟡 Serious Defect",
                "desc": "Failure of the weld metal to extend fully through the joint thickness or weld root, leaving an unbonded gap.",
                "action": "SERIOUS ACTION REQUIRED: Lack of penetration detected. This significantly reduces weld throat thickness and load-bearing capacity. Requires mechanical grinding of the root and re-welding to ensure complete penetration."
            },
            "PO": {
                "name": "Porosity (PO)",
                "level": "WARNING",
                "color": "#F97316",  # Safety Orange
                "bg_color": "rgba(249, 115, 22, 0.1)",
                "border_color": "#F97316",
                "badge": "🟠 Warning / Moderate",
                "desc": "Gas cavities or bubbles trapped within the weld pool during solidification, typically caused by moisture, contaminants, or inadequate shielding gas flow.",
                "action": "MODERATE SUSPICION: Trapped gas cavities detected. Inspect shield gas flow rate, nozzle cleanliness, and ensure electrodes are dry. If porosity density exceeds structural quality standards (e.g. ISO 5817), grind out the porous region and re-weld."
            },
            "ND": {
                "name": "No Defect (ND)",
                "level": "PASS",
                "color": "#10B981",  # Emerald Green
                "bg_color": "rgba(16, 185, 129, 0.1)",
                "border_color": "#10B981",
                "badge": "🟢 Quality Pass",
                "desc": "A sound weld joint that shows no structurally significant internal or external defects. Conforms to baseline radiographic standards.",
                "action": "STRUCTURAL CONFORMANCE: Weld conforms to design specifications. No cracks, lack of penetration, or significant porosity detected. Clear the joint for normal surface treatment and subsequent load assembly."
            }
        }

    def _resolve_input(self, image_path):
        """
        Returns (path_to_infer, is_temp). When apply_preprocessing is on, applies
        the CLAHE + denoise pipeline and writes the result to a temp file so the
        model sees the same distribution it was trained on. Caller must delete the
        temp file when is_temp is True.
        """
        if not self.apply_preprocessing:
            return image_path, False
        import cv2
        import tempfile
        from preprocess import preprocess_image
        processed = preprocess_image(image_path)
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(tmp_path, processed)
        return tmp_path, True

    def predict(self, image_path):
        """
        Runs model inference on the provided image path.
        Returns a dict with class predictions, confidence, descriptions, and custom theme details.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target image not found at: {image_path}")

        infer_path, is_temp = self._resolve_input(image_path)
        try:
            results = self.model(infer_path, verbose=False)
        finally:
            if is_temp:
                try:
                    os.remove(infer_path)
                except OSError:
                    pass
        result = results[0]

        # Parse probabilities
        probs = result.probs
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        
        # Get names dictionary from model, fall back to default map
        model_names = getattr(result, "names", {0: "CR", 1: "LP", 2: "PO", 3: "ND"})
        top_class_code = model_names.get(top1_idx, "ND")
        
        # Get class metadata
        details = self.class_details.get(top_class_code, self.class_details["ND"])
        
        # Format prediction dictionary
        prediction = {
            "class_code": top_class_code,
            "class_name": details["name"],
            "confidence": top1_conf,
            "level": details["level"],
            "color": details["color"],
            "bg_color": details["bg_color"],
            "border_color": details["border_color"],
            "badge": details["badge"],
            "description": details["desc"],
            "recommendation": details["action"],
            "all_probabilities": {}
        }
        
        # Include all class probabilities
        probs_data = probs.data.tolist()
        for idx, score in enumerate(probs_data):
            class_code = model_names.get(idx, f"CLASS_{idx}")
            class_name = self.class_details.get(class_code, {}).get("name", class_code)
            prediction["all_probabilities"][class_name] = float(score)
        return prediction

    def generate_gradcam(self, image_path, output_path=None):
        """
        Generates a Grad-CAM heatmap overlay for the predicted class.
        If output_path is provided, saves it there.
        Returns the overlayed image as a numpy array in BGR format.
        """
        import cv2
        import numpy as np
        
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
        except ImportError:
            raise ImportError(
                "The 'pytorch-grad-cam' library is required for explainable AI heatmaps. "
                "Please install it using: pip install grad-cam"
            )
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target image not found at: {image_path}")
            
        # Run inference first to get prediction details (handles preprocessing internally)
        pred = self.predict(image_path)
        class_code = pred["class_code"]
        confidence = pred["confidence"]

        # Mapping class code to index (CR:0, LP:1, PO:2, ND:3)
        class_idx_map = {"CR": 0, "LP": 1, "PO": 2, "ND": 3}
        target_idx = class_idx_map.get(class_code, 3)

        # Load and resize the SAME image the model sees (preprocessed for v2) so the
        # heatmap aligns with the activations.
        infer_path, is_temp = self._resolve_input(image_path)
        try:
            image = cv2.imread(infer_path)
        finally:
            if is_temp:
                try:
                    os.remove(infer_path)
                except OSError:
                    pass
        if image is None:
            raise ValueError(f"Could not load image at: {image_path}")
        image_resized = cv2.resize(image, (224, 224))
        rgb_img = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB) / 255.0

        # Convert to torch tensor
        input_tensor = torch.from_numpy(rgb_img.transpose(2, 0, 1)).unsqueeze(0).float()
        
        # Wrap model and enable grad on parameters
        wrapper = YOLOClassifierWrapper(self.model)
        for param in wrapper.pytorch_model.parameters():
            param.requires_grad = True
            
        wrapper.eval()
        
        # Target layer is the last convolutional block before classification head
        target_layers = [wrapper.pytorch_model.model[-2]]
        
        # Generate GradCAM activations
        with torch.enable_grad():
            cam = GradCAM(model=wrapper, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
            
        # Overlay heatmap
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        output_img = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        
        # Add labels on the image
        label_text = f"{pred['class_name']} ({confidence*100:.1f}%)"
        cv2.putText(
            output_img, 
            label_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 0), 
            2, 
            cv2.LINE_AA
        )
        
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, output_img)
            
        return output_img



if __name__ == "__main__":
    # Test script if run directly
    try:
        predictor = WeldPredictor()
        print("Model loaded successfully.")
        print(f"Supported classes: {list(predictor.class_details.keys())}")
    except Exception as e:
        print(f"Error during model loading: {e}")
