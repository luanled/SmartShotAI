"""
Training script for EMOv2 on the AVA dataset using Earth Mover's Distance (EMD) Loss.
step1 freeze backbone: python train_ava.py --ava_root ..\data\AVA_dataset --variant emo2_5m_224 --epochs 15 --batch_size 112 --lr 3e-4 --freeze_backbone --amp --out_dir runs\ava_frozen
step2 Fine-tuning: python train_ava.py --ava_root ..\data\AVA_dataset --variant emo2_5m_224 --epochs 30 --batch_size 112 --lr 1e-4 --lr_backbone 1e-5 --resume runs\ava_frozen\ckpt_best.pt --amp --out_dir runs\ava_finetuned
"""

import os
from PIL import ImageFile # avoid image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

# Force remove SSL to avoid writing permission issues
if "SSLKEYLOGFILE" in os.environ:
    del os.environ["SSLKEYLOGFILE"]

# --- our modules ---
from models.emo_v2_official import EMOBackboneRegressor as StudentBackbone
from datasets.ava import AVADataset
from utils.metrics import eval_regression
from utils.common import seed_all, get_device, AvgMeter, save_ckpt, load_ckpt, auto_num_workers


# ---------- EMD Loss ----------
class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, p_pred, p_true):
        """
        Calculates the Squared Earth Mover's Distance.
        p_pred: [BatchSize, 10] Predicted probability distribution
        p_true: [BatchSize, 10] Ground truth probability distribution
        """
        cdf_pred = torch.cumsum(p_pred, dim=1)
        cdf_true = torch.cumsum(p_true, dim=1)

        emd_loss = torch.mean((cdf_pred - cdf_true) ** 2, dim=1)
        return torch.mean(emd_loss)


# ---------- helpers ----------
def build_transforms(img_size, mean, std, train: bool):
    if train:
        tx = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        tx = T.Compose([
            T.Resize(int(img_size / 0.875)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return tx


@torch.no_grad()
def evaluate_loop(model, loader, device):
    model.eval()
    ys, yh = [], []

    # Weights for calculating the expected score [1.0, 2.0, ..., 10.0]
    score_weights = torch.arange(1, 11, device=device, dtype=torch.float32)

    for xb, yb, _ in tqdm(loader, desc="eval", leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # p_pred shape: [B, 10]
        ph_dist = model(xb)

        # Calculate expected mean score
        ph_mean = torch.sum(ph_dist * score_weights, dim=1)
        yb_mean = torch.sum(yb * score_weights, dim=1)

        yh.extend(ph_mean.cpu().numpy().tolist())
        ys.extend(yb_mean.cpu().numpy().tolist())

    out = {"GT": eval_regression(ys, yh)}
    return out


def make_loaders(args, student):
    img_size = student.input_size
    mean, std = student.norm_mean, student.norm_std

    ds_train = AVADataset(args.ava_root, split="train", resize=img_size)
    ds_val = AVADataset(args.ava_root, split="val", resize=img_size)

    ds_train.tx = build_transforms(img_size, mean, std, train=True)
    ds_val.tx = build_transforms(img_size, mean, std, train=False)

    nw = auto_num_workers() if args.num_workers < 0 else args.num_workers
    persistent = bool(nw > 0)

    train_ld = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=persistent
    )
    val_ld = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=persistent
    )
    return train_ld, val_ld, img_size, mean, std


def save_metrics_to_disk(out_dir, epoch, metrics, is_best, best_srcc):
    val_file = os.path.join(out_dir, "val_metrics.json")
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump({"epoch": epoch, **metrics}, f, indent=2)

    events_file = os.path.join(out_dir, "events.jsonl")
    with open(events_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": "val", "epoch": epoch, **metrics}) + "\n")

    epoch_file = os.path.join(out_dir, f"val_metrics_epoch_{epoch:03d}.json")
    with open(epoch_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if is_best:
        best_file = os.path.join(out_dir, "val_metrics_best.json")
        with open(best_file, "w", encoding="utf-8") as f:
            json.dump({"epoch": epoch, "SRCC": best_srcc, **metrics}, f, indent=2)

        with open(events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": "val_best", "epoch": epoch, "SRCC": best_srcc, **metrics}) + "\n")


# ---------- main ----------
def parse_args():
    ap = argparse.ArgumentParser("EMOv2 student KD with AVA Dataset and EMD Loss")

    # --- Data ---
    ap.add_argument("--ava_root", type=str, default="../data/AVA_dataset", help="Path to AVA root")

    # --- Loader ---
    ap.add_argument("--num_workers", type=int, default=-1, help="-1=auto; else fixed workers")
    ap.add_argument("--batch_size", type=int, default=112)

    # --- Student (EMOv2) ---
    ap.add_argument("--variant", type=str, default="emo2_5m_224")
    ap.add_argument("--head_hidden", type=int, default=256)

    # --- Train ---
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)  # for head
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=295)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--early_stop_patience", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="runs/ava_debug")

    # --- Fine-tune ---
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--freeze_backbone", action="store_true")

    args = ap.parse_args()
    return args


def main():
    args = parse_args()
    seed_all(args.seed)
    device = get_device()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    log_path = os.path.join(args.out_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting AVA training with args: {vars(args)}")

    # --- student backbone ---
    student = StudentBackbone(variant=args.variant, head_hidden=args.head_hidden).to(device)

    # Dynamically modify the head to output 10 probabilities for AVA
    in_features = student.head[2].in_features
    student.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(in_features, args.head_hidden),
        nn.SiLU(),
        nn.Linear(args.head_hidden, 10),
        nn.Softmax(dim=1)
    ).to(device)

    # --- data loaders ---
    train_ld, val_ld, img_size, mean, std = make_loaders(args, student)

    # --- optim & loss ---
    if args.freeze_backbone:
        logger.info("Freezing backbone parameters. Only training the head.")
        for param in student.backbone.parameters():
            param.requires_grad = False
        opt_params = student.head.parameters()
    else:
        logger.info(f"Using differential LR: Backbone={args.lr_backbone}, Head={args.lr}")
        for param in student.backbone.parameters():
            param.requires_grad = True
        opt_params = [
            {"params": student.backbone.parameters(), "lr": args.lr_backbone},
            {"params": student.head.parameters(), "lr": args.lr}
        ]

    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    # Use EMD Loss for AVA distribution learning
    criterion_emd = EMDLoss().to(device)

    best_srcc = -1.0
    epochs_no_improve = 0
    start_epoch = 1

    if getattr(args, "resume", None):
        if os.path.isfile(args.resume):
            try:
                ck = load_ckpt(args.resume, model=student, optim=opt, map_location="cpu")
                logger.info("Successfully loaded model weights and optimizer state.")
            except Exception as e:
                logger.warning(f"Optimizer state mismatch. Falling back to weight-only load. Reason: {e}")
                ck = load_ckpt(args.resume, model=student, optim=None, map_location="cpu")
                logger.info("Successfully loaded model weights only.")

            extra = ck.get("extra", {})
            metrics_prev = extra.get("metrics", {})
            best_srcc = metrics_prev.get("GT", {}).get("SRCC", -1.0)
            if best_srcc is None:
                best_srcc = -1.0
            start_epoch = int(extra.get("epoch", 0)) + 1
            logger.info(f"Resumed from {args.resume} | start_epoch={start_epoch} | best_SRCC={best_srcc:.4f}")
        else:
            logger.warning(f"Resume file not found: {args.resume}")

    # --- training loop ---
    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        meter = AvgMeter()
        pbar = tqdm(train_ld, desc=f"Epoch {epoch}/{args.epochs}")

        for xb, yb, _ in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                yh = student(xb)
                loss = criterion_emd(yh, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            meter.update(loss.item(), xb.size(0))
            pbar.set_postfix(loss=f"{meter.avg:.4f}")

        # --- validation ---
        metrics = evaluate_loop(student, val_ld, device)

        srcc_now = metrics.get("GT", {}).get("SRCC", -1.0)
        is_best = srcc_now > best_srcc + 1e-4

        if is_best:
            best_srcc = srcc_now
            epochs_no_improve = 0
            logger.info(f"[VAL] Epoch {epoch}: New Best GT SRCC = {best_srcc:.4f} (Improved!)")
        else:
            epochs_no_improve += 1
            logger.info(
                f"[VAL] Epoch {epoch}: GT SRCC = {srcc_now:.4f} (No improvement: {epochs_no_improve}/{args.early_stop_patience})")

        save_metrics_to_disk(args.out_dir, epoch, metrics, is_best, best_srcc)

        save_ckpt(
            os.path.join(args.out_dir, "ckpt_last.pt"),
            student, opt, epoch,
            extra={"metrics": metrics, "epoch": epoch, "variant": args.variant}
        )

        if is_best:
            save_ckpt(
                os.path.join(args.out_dir, "ckpt_best.pt"),
                student, opt, epoch,
                extra={"metrics": metrics, "epoch": epoch, "variant": args.variant}
            )

        if epochs_no_improve >= args.early_stop_patience:
            logger.warning(f"[Early Stop] Stopping training due to no SRCC improvement.")
            break

    logger.info(f"Training Complete! Overall Best SRCC: {best_srcc:.4f}")


if __name__ == "__main__":
    main()