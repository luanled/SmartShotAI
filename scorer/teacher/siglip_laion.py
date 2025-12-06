import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoImageProcessor, SiglipVisionModel

# Choose HF SigLIP vision variant; base-16 (224).
_DEFAULT_BACKBONE = "google/siglip-base-patch16-224"

class AestheticHead(nn.Module):
    """Simple MLP head on top of frozen SigLIP embeddings -> scalar [0,1]."""
    def __init__(self, embed_dim=768, hidden=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.out = nn.Sigmoid()

    def forward(self, x):
        return self.out(self.mlp(x)).squeeze(1)

class SigLipTeacher(nn.Module):
    """
    Teacher = frozen SigLIP image backbone + aesthetics MLP head.
    Provides:
      - get_backbone(): (nn.Module, embed_dim, processor)
      - encode(images): pooled embeddings from the backbone
      - forward(images): final aesthetic scores in [0,1]
    """
    def __init__(self, model_name=_DEFAULT_BACKBONE, head_ckpt=None, device="cuda", freeze_backbone=True):
        super().__init__()
        self.device = torch.device(device)

        # HuggingFace preprocessor (handles resize/center-crop/normalize internally)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        # --- BACKBONE (this IS the "backbone") ---
        self.backbone = SiglipVisionModel.from_pretrained(model_name).to(self.device)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.backbone.eval()

        # Dim depends on the chosen model (e.g., 768 for base, >768 for larger)
        self.embed_dim = int(self.backbone.config.hidden_size)

        # --- HEAD (aesthetics) ---
        self.head = AestheticHead(embed_dim=self.embed_dim).to(self.device)
        if head_ckpt is not None:
            state = torch.load(head_ckpt, map_location="cpu")
            # accept state dict or wrapper {"model": state_dict}
            if "state_dict" in state: state = state["state_dict"]
            if "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            self.head.load_state_dict(state, strict=False)

        # Optional torchvision-style resize (if you pass raw PIL/tensors directly)
        self.tx = transforms.Compose([transforms.Resize((224, 224))])

        self.eval()

    # ---------- helpers ----------
    def get_backbone(self):
        """
        Returns the underlying SigLIP vision model, embedding dim, and HF processor.
        Useful if you want to re-use/call the encoder elsewhere.
        """
        return self.backbone, self.embed_dim, self.processor

    @torch.no_grad()
    def encode(self, images, return_tokens=False):
        """
        images: float tensor in [0,1], shape [B,3,H,W]
        returns: pooled embeddings [B, D] (or token embeddings if return_tokens=True)
        """
        # Convert batch of tensors to the HF processor format
        inputs = self.processor(
            images=[(img * 255).byte().permute(1, 2, 0).cpu().numpy() for img in images],
            return_tensors="pt"
        ).to(self.device)

        out = self.backbone(**inputs)
        if return_tokens:
            # [B, num_patches+1, D] (includes class token)
            return out.last_hidden_state
        # default pooled embedding [B, D]
        return out.pooler_output

    # ---------- main forward ----------
    @torch.no_grad()
    def forward(self, images):
        """
        images: float tensor in [0,1], shape [B,3,H,W]
        returns: scores in [0,1], shape [B]
        """
        feats = self.encode(images, return_tokens=False)  # [B, D]
        scores = self.head(feats)                          # [B]
        return scores
