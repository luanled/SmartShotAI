import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from PIL import Image

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SigLIPAestheticScorer(nn.Module):
    def __init__(self, model_name="google/siglip-base-patch16-224"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name).vision_model
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Flatten(0) 
        )

    def forward(self, x):
        outputs = self.backbone(x)
        return self.head(outputs.pooler_output)

def predict_aesthetic_score(image_path, model_path="siglip_tad66k_best.pt"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading model on {device}...")

    model_name = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    model = SigLIPAestheticScorer(model_name).to(device)

    # Load your saved weights!
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Analyzing: {image_path}")
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)

    with torch.no_grad():
        score = model(pixel_values)
        
    final_score = score.item()
    print("-" * 30)
    print(f"Predicted Aesthetic Score: {final_score:.2f} / 10.0")
    print("-" * 30)

if __name__ == "__main__":
    # CHANGE THIS to the path of any image you want to test!
    TEST_IMAGE = "test_photo.jpg" 
    predict_aesthetic_score(TEST_IMAGE)