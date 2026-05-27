from ultralytics import YOLO
import os
import torch

class WeldPredictor:
    def __init__(self, model_path=None):
        """
        Initializes the Weld Defect Predictor.
        Default path points to the best weights trained.
        """
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

    def predict(self, image_path):
        """
        Runs model inference on the provided image path.
        Returns a dict with class predictions, confidence, descriptions, and custom theme details.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target image not found at: {image_path}")
            
        results = self.model(image_path, verbose=False)
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

if __name__ == "__main__":
    # Test script if run directly
    try:
        predictor = WeldPredictor()
        print("Model loaded successfully.")
        print(f"Supported classes: {list(predictor.class_details.keys())}")
    except Exception as e:
        print(f"Error during model loading: {e}")
