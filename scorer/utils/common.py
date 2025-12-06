# utils/common.py
import os, platform, random, torch, numpy as np

class AvgMeter:
    def __init__(self): self.reset()
    def reset(self): self.n=0; self.sum=0.0
    def update(self, v, k=1): self.sum += v*k; self.n += k
    @property
    def avg(self): return self.sum / max(self.n, 1)

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_ckpt(path, model, optim, step, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "step": step,
        "extra": extra or {},
    }, path)

def load_ckpt(path, model=None, optim=None, map_location="cpu"):
    obj = torch.load(path, map_location=map_location)
    if model is not None: model.load_state_dict(obj["model"], strict=False)
    if optim is not None and "optim" in obj: optim.load_state_dict(obj["optim"])
    return obj

def auto_num_workers():
    """Heuristic that works well on Windows desktops and Jetson."""
    cores = os.cpu_count() or 4
    system = platform.system().lower()
    # Detect Jetson (very rough)
    is_jetson = os.path.exists("/proc/device-tree/model") and \
                "jetson" in open("/proc/device-tree/model","r",errors="ignore").read().lower()
    if is_jetson:
        return 2
    if "windows" in system:
        return min(4, max(0, cores // 4))   # 1 worker per ~4 cores, cap 4
    # Linux desktop/server
    return min(8, max(2, cores // 2))
