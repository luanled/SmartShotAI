import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image

# pip install aesthetic-predictor-v2-5
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

class V25Teacher(nn.Module):
    """
    Wraps the public SigLIP-based Aesthetic Predictor V2.5
    Usage: scores = teacher(images_01)  # [B,3,H,W] in [0,1] -> [B] in [0,1]
    """
    def __init__(self, device="cuda", low_cpu_mem_usage=True, trust_remote_code=True):
        super().__init__()
        self.device = torch.device(device)
        # model(pixel_values).logits ≈ 1..10 ; preproc handles resize/normalize
        self.model, self.preproc = convert_v2_5_from_siglip(
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=trust_remote_code
        )
        self.model.to(self.device).eval()

    @torch.no_grad()
    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        """
        images_01: float tensor in [0,1], shape [B,3,H,W]
        returns: scores in [0,1], shape [B]
        """
        # convert each tensor -> PIL (preproc expects PIL in most setups)
        pils = [to_pil_image(img.clamp(0,1).cpu()) for img in images_01]
        # try HF-style processor first
        try:
            batch = self.preproc(images=pils, return_tensors="pt")
            pixel_values = batch["pixel_values"].to(self.device)
        except TypeError:
            # if preproc is a torchvision transform: map over list
            pixel_values = torch.stack([self.preproc(p) for p in pils]).to(self.device)

        out = self.model(pixel_values).logits  # [B] or [B,1], ~1..10
        if out.ndim > 1:
            out = out.squeeze(-1)
        # map 1..10 → [0,1] (monotonic; good for KD/ranking)
        scores01 = (out - 1.0) / 9.0
        return scores01.clamp(0.0, 1.0)
