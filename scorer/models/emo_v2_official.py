import os, sys, torch, torch.nn as nn

# 1) Make the vendored repo importable
BASE_DIR = os.path.dirname(__file__)                         # .../smartPhoto/models
SMARTPHOTO_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))       # .../smartPhoto
CANDIDATES = [
    os.path.join(SMARTPHOTO_DIR, "third_party", "EMOv2-main"),       # NEW location
    os.path.join(os.path.abspath(os.path.join(SMARTPHOTO_DIR, "..")), "third_party", "EMOv2-main"),  # old fallback
    os.environ.get("EMO_ROOT", ""),                                   # optional env override
]
for p in CANDIDATES:
    if p and os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
        break

# 2) Import EMO2 builders from the repo
from model.lib_emo.emo2 import (  # from your repo
    EMO2_5M_k5_hybrid, EMO2_5M_k5_hybrid_256, EMO2_5M_k5_hybrid_512,
    EMO2_2M_k5_hybrid, EMO2_2M_k5_hybrid_256, EMO2_2M_k5_hybrid_512,
    EMO2_1M_k5_hybrid, EMO2_1M_k5_hybrid_256, EMO2_1M_k5_hybrid_512,
)

_VARIANTS = {
    "emo2_1m_224": (EMO2_1M_k5_hybrid, 224),
    "emo2_1m_256": (EMO2_1M_k5_hybrid_256, 256),
    "emo2_1m_512": (EMO2_1M_k5_hybrid_512, 512),
    "emo2_2m_224": (EMO2_2M_k5_hybrid, 224),
    "emo2_2m_256": (EMO2_2M_k5_hybrid_256, 256),
    "emo2_2m_512": (EMO2_2M_k5_hybrid_512, 512),
    "emo2_5m_224": (EMO2_5M_k5_hybrid, 224),
    "emo2_5m_256": (EMO2_5M_k5_hybrid_256, 256),
    "emo2_5m_512": (EMO2_5M_k5_hybrid_512, 512),
}

class EMOBackboneRegressor(nn.Module):
    """Use EMO2 as a backbone; add a small head to regress an aesthetic score in [0,1]."""
    def __init__(self, variant="emo2_5m_224", head_hidden=256):
        super().__init__()
        builder, img_size = _VARIANTS[variant]
        # note: num_classes is ignored; we'll use forward_features()
        self.backbone = builder(pretrained=False, num_classes=1000)
        self.input_size = img_size
        # ImageNet mean/std from the EMO2 configs
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std  = [0.229, 0.224, 0.225]

        # infer feature dim by a dry forward (safe, one-time)
        with torch.no_grad():
            x = torch.zeros(1, 3, img_size, img_size)
            f = self.backbone.forward_features(x)  # [1, C, H, W]
            c = f.shape[1]

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, head_hidden), nn.SiLU(),
            nn.Linear(head_hidden, 1), nn.Sigmoid()
        )

    def forward(self, x):
        f = self.backbone.forward_features(x)  # features (no classifier)
        return self.head(f).squeeze(1)