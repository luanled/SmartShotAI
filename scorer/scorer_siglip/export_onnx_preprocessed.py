import torch
import torch.nn as nn
import torch.nn.functional as F
from train_siglip_ava import SigLIPAestheticScorer

# -----------------------------
# 1. Preprocessing Module
# -----------------------------
class Preprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

    def forward(self, x):
        # x: (1, H, W, 3) uint8 or float32
        x = x.float() / 255.0                     # scale to 0–1
        x = x.permute(0, 3, 1, 2)                 # HWC → CHW
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std            # normalize
        return x

# -----------------------------
# 2. Combined Model
# -----------------------------
class WrappedModel(nn.Module):
    def __init__(self, core_model):
        super().__init__()
        self.pre = Preprocess()
        self.model = core_model

    def forward(self, x):
        x = self.pre(x)
        return self.model(x)

# -----------------------------
# 3. Load your trained model
# -----------------------------
core_model = SigLIPAestheticScorer()
state = torch.load("siglip_ava_finetuned.pt", map_location="cpu")
core_model.load_state_dict(state)
core_model.eval()

model = WrappedModel(core_model).eval()

# -----------------------------
# 4. Dummy input (raw image)
# -----------------------------
dummy = torch.randint(0, 255, (1, 224, 224, 3), dtype=torch.uint8)

# -----------------------------
# 5. Export ONNX
# -----------------------------
torch.onnx.export(
    model,
    dummy,
    "siglip_aesthetic_preprocessed.onnx",
    input_names=["raw_image"],
    output_names=["score"],
    opset_version=17,
    dynamic_axes=None,
)

print("Saved siglip_aesthetic_preprocessed.onnx")
