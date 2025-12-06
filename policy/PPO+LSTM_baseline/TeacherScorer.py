import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoModel

class TeacherScorer:
    def __init__(self, mode="heuristic", weights_path="siglip_head_best.pth"):
        """
        mode: 'heuristic' (fast, CV-based) or 'deep' (slow, SigLIP-based)
        weights_path: Path to the .pth file saved from your training notebook.
        """
        self.mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Apple Metal (MPS) support for Mac users
        if torch.backends.mps.is_available():
            self.device = "mps"
            
        print(f"Initializing TeacherScorer (Mode: {mode}) on {self.device}...")
        
        if self.mode == 'deep':
            # 1. Load the Processor and Base Model (Frozen)
            # We use the standard pre-trained SigLIP from Google
            self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
            self.base_model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(self.device)
            self.base_model.eval()

            # 2. Re-create the Head Architecture
            hidden_size = self.base_model.config.vision_config.hidden_size
            self.head = nn.Linear(hidden_size, 1).to(self.device)
            
            # 3. Load the Trained Weights
            try:
                # Load weights to CPU first to avoid mapping issues, then move to device
                state_dict = torch.load(weights_path, map_location=self.device)
                self.head.load_state_dict(state_dict)
                self.head.eval()
                print(f"✅ Successfully loaded SigLIP weights from {weights_path}")
            except FileNotFoundError:
                print(f"❌ Error: Could not find '{weights_path}'. Please move the .pth file here.")
                # Fallback to avoid crash, but warn user
                self.mode = "heuristic" 
                print("⚠️ Falling back to Heuristic mode.")

    def get_score(self, frame):
        """
        Input: frame (H, W, 3) numpy array, uint8, RGB
        Output: scalar float 0.0 - 1.0
        """
        if self.mode == 'heuristic':
            return self._calculate_heuristic_score(frame)
        else:
            return self._calculate_deep_score(frame)

    def _calculate_deep_score(self, frame):
        # 1. Convert Numpy (H,W,C) -> PIL Image
        pil_image = Image.fromarray(frame)

        # 2. Preprocess (Normalize & Resize)
        # The processor handles the [0,1] normalization internally
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        # 3. Inference
        with torch.no_grad():
            features = self.base_model.get_image_features(**inputs) 
            
            raw_score = self.head(features).item()
            
        # 4. Normalize Output
        # The training data (AVA) was 1-10. We need 0-1.
        # We assume the model predicts roughly in the 1-10 range.
        normalized_score = raw_score / 10.0
        
        # Clamp to ensure strict [0,1] range compliance
        return float(np.clip(normalized_score, 0.0, 1.0))

    def _calculate_heuristic_score(self, frame):
        """
        A fast proxy for aesthetics: prefers bright, high-contrast, colorful images.
        """
        # 1. Convert to HSV for colorfulness
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].mean() / 255.0
        
        # 2. Calculate contrast (standard deviation of brightness)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        contrast = gray.std() / 128.0 
        
        # 3. Brightness 
        brightness = gray.mean() / 255.0
        
        # Composite score
        score = (0.4 * saturation) + (0.3 * contrast) + (0.3 * brightness)
        
        return float(np.clip(score, 0.0, 1.0))