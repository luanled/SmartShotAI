import torch
import coremltools as ct
from train_siglip_ava import SigLIPAestheticScorer

# 1. Load model architecture
model = SigLIPAestheticScorer()

# 2. Load your BEST AVA-finetuned weights
state = torch.load("siglip_ava_finetuned.pt", map_location="cpu")
model.load_state_dict(state)

model.eval()

# 3. Example input
example = torch.randn(1, 3, 224, 224)

# 4. Trace
traced = torch.jit.trace(model, example)

# 5. Convert to CoreML
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(shape=example.shape)],
    compute_units=ct.ComputeUnit.ALL,
)

# 6. Save as ML Program
mlmodel.save("siglip_aesthetic.mlpackage")
print("Saved siglip_aesthetic.mlpackage")
