# utils/metrics.py
import numpy as np
from scipy.stats import spearmanr, pearsonr
import torch

def srcc(y_true, y_pred):
    return float(spearmanr(y_true, y_pred).correlation)

def plcc(y_true, y_pred):
    return float(pearsonr(y_true, y_pred)[0])

def rmse(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def eval_regression(y_true, y_pred):
    return {"SRCC": srcc(y_true, y_pred),
            "PLCC": plcc(y_true, y_pred),
            "RMSE": rmse(y_true, y_pred)}

"""
# call it as eval vs teacher later
metrics = validate(student, val_loader, device, teacher if args.use_teacher else None)
print(metrics)

"""
@torch.no_grad()
def validate(model, loader, device, teacher=None):
    model.eval()
    ys, yh, yt = [], [], []
    for x, y, _ in loader:
        x = x.to(device)
        yh.extend(model(x).cpu().numpy().tolist())
        if teacher is not None:
            t = teacher((x * 0.5 + 0.5).clamp(0,1)).cpu().numpy().tolist()
            yt.extend(t)
        ys.extend(y.numpy().tolist())
    out = {"GT": eval_regression(ys, yh)}
    if teacher is not None:
        out["Teacher"] = eval_regression(yt, yh)
    return out
