"""
# changed timm import in emov2-model __init__.py and emo2.py with opffical env timm==0.6.5
  For run in aadb: python train_distill_emo.py --aadb_root ../data/AADB --variant emo2_5m_224 --epochs 10 --batch_size 32 --amp --use_teacher --teacher_kind v25 --out_dir runs/aadb_kd_v25
  For run in tad66k python train_distill_emo.py --dataset tad66k --tad66k_root ../data/TAD66K --variant emo2_5m_224 --epochs 2 --batch_size 32 --amp --use_teacher --teacher_kind v25 --out_dir runs/tad66k_kd_v25
  For resume: python train_distill_emo.py --aadb_root ..\data\AADB --img_subdir datasetImages_originalSize --variant emo2_5m_224 --epochs 30 --early_stop_patience 2 --num_workers -1 --use_teacher --teacher_kind v25 --resume runs\aadb_kd_v25\ckpt_best.pt --out_dir runs\aadb_kd_v25
"""
"""
"GT" = correlation/RMSE between student predictions and AADB ground-truth labels.
"Teacher" = correlation/RMSE between student predictions and the teacher outputs.
saving the best checkpoint using the “Teacher SRCC”

Epoch 1/1: 100%|██████████| 1254/1254 [1:37:55<00:00,  4.69s/it, loss=0.0053]
[VAL] {
  "GT": {
    "SRCC": 0.03579494604675367,
    "PLCC": 0.035946320722603176,
    "RMSE": 0.3579439898861815
  },
  "Teacher": {
    "SRCC": 0.1074333819930145,
    "PLCC": 0.10731116240405891,
    "RMSE": 0.07111320313192801
  }
}
Done. Best Teacher SRCC: 0.1074
the evaluation set: 
train test split
metrics, threshold
table better?
"""

"""

[VAL] {
  "GT": {
    "SRCC": 0.12739465331676567,
    "PLCC": 0.134852591128314,
    "RMSE": 0.7835750151052484
  },
  "Teacher": {
    "SRCC": 0.4114168485513826,
    "PLCC": 0.43214794575368326,
    "RMSE": 0.06815407224675823
  }
}
[save] new best SRCC=0.4114
Done. Best SRCC: 0.4114
"""

import os, argparse, json, math, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

# --- our modules ---
from models.emo_v2_official import EMOBackboneRegressor as StudentBackbone
from teacher.siglip_laion import SigLipTeacher
from datasets.aadb import AADBDataset  # we will override its .tx after construction
from datasets.tad66k import TAD66KDataset
from utils.metrics import eval_regression
from utils.common import seed_all, get_device, AvgMeter, save_ckpt, load_ckpt, auto_num_workers

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
        # mirrors EMO2 test pipeline in configs: Resize -> CenterCrop -> Normalize
        tx = T.Compose([
            T.Resize(int(img_size / 0.875)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return tx

@torch.no_grad()
def denorm_to_01(x, mean, std):
    """
    x: normalized tensor [B,3,H,W] with given mean/std -> back to [0,1]
    """
    mean = torch.as_tensor(mean, device=x.device).view(1, -1, 1, 1)
    std  = torch.as_tensor(std,  device=x.device).view(1, -1, 1, 1)
    y = x * std + mean
    return y.clamp(0.0, 1.0)

@torch.no_grad()
def evaluate_loop(model, loader, device, teacher=None, mean=None, std=None):
    model.eval()
    ys, yh, yt = [], [], []
    for xb, yb, _ in tqdm(loader, desc="eval", leave=False):
        xb = xb.to(device, non_blocking=True)
        ph = model(xb).cpu().numpy().tolist()
        yh.extend(ph)
        ys.extend(yb.numpy().tolist())
        if teacher is not None:
            tb = teacher(denorm_to_01(xb, mean, std)).cpu().numpy().tolist()
            yt.extend(tb)
    out = {"GT": eval_regression(ys, yh)}
    if teacher is not None and len(yt) == len(yh):
        out["Teacher"] = eval_regression(yt, yh)
    return out

# ---------- main ----------
def parse_args():
    import argparse, os
    ap = argparse.ArgumentParser("EMOv2 student KD with SigLIP+LAION teacher")

    # --- Data ---
    ap.add_argument("--dataset", type=str, default="aadb",
                    choices=["aadb", "tad66k"], help="dataset key")
    ap.add_argument("--aadb_root", type=str, default="../data/AADB",
                    help="Path to AADB root (required for --dataset aadb)")
    ap.add_argument("--tad66k_root", type=str, default="../data/TAD66K",
                    help="Path to TAD66K root (required for --dataset tad66k)")
    ap.add_argument("--img_subdir", type=str, default="datasetImages_originalSize",
                    help="AADB images subdir; ignored for TAD66K")

    # --- Loader ---
    ap.add_argument("--num_workers", type=int, default=-1, help="-1=auto; else fixed workers")
    ap.add_argument("--batch_size", type=int, default=32)

    # --- Student (EMOv2) ---
    ap.add_argument("--variant", type=str, default="emo2_5m_224",
                    choices=["emo2_1m_224","emo2_1m_256","emo2_1m_512",
                             "emo2_2m_224","emo2_2m_256","emo2_2m_512",
                             "emo2_5m_224","emo2_5m_256","emo2_5m_512"])
    ap.add_argument("--head_hidden", type=int, default=256)

    # --- Teacher ---
    ap.add_argument("--use_teacher", action="store_true")
    ap.add_argument("--teacher_kind", type=str, default="v25",
                    choices=["v25","siglip_mlp"])
    ap.add_argument("--teacher_model", type=str, default=None)
    ap.add_argument("--teacher_head_ckpt", type=str, default=None)

    # --- Train ---
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=295)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--early_stop_patience", type=int, default=8)
    ap.add_argument("--lambda_distill", type=float, default=1.0)
    ap.add_argument("--lambda_supervised", type=float, default=0.0)
    ap.add_argument("--out_dir", type=str, default="runs/debug")

    # ---- Parse once ----
    args = ap.parse_args()

    # ---- Conditional validation ----
    if args.dataset == "aadb":
        if not args.aadb_root:
            ap.error("--aadb_root is required when --dataset=aadb")
        if not os.path.isdir(args.aadb_root):
            ap.error(f"--aadb_root not found: {args.aadb_root}")

    if args.dataset == "tad66k":
        if not args.tad66k_root:
            ap.error("--tad66k_root is required when --dataset=tad66k")
        if not os.path.isdir(args.tad66k_root):
            ap.error(f"--tad66k_root not found: {args.tad66k_root}")

    return args

def make_loaders(args, student):
    # build transforms from student’s preprocessing contract
    img_size = student.input_size
    mean, std = student.norm_mean, student.norm_std

    # datasets
    if args.dataset == "aadb":
        ds_train = AADBDataset(args.aadb_root, split="train",
                               img_subdir=args.img_subdir, resize=img_size)
        ds_val = AADBDataset(args.aadb_root, split="val",
                             img_subdir=args.img_subdir, resize=img_size, center_crop=True)
    elif args.dataset == "tad66k":
        ds_train = TAD66KDataset(args.tad66k_root, split="train", resize=img_size)
        ds_val = TAD66KDataset(args.tad66k_root, split="val", resize=img_size, center_crop=True)

    # transforms (match EMO2 policy)
    ds_train.tx = build_transforms(img_size, mean, std, train=True)
    ds_val.tx   = build_transforms(img_size, mean, std, train=False)

    # --- handle -1 as "auto" and set persistent_workers safely
    nw = auto_num_workers() if args.num_workers < 0 else args.num_workers
    persistent = bool(nw > 0)

    train_ld = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=persistent
    )
    val_ld = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=persistent
    )
    return train_ld, val_ld, img_size, mean, std

def main():
    args = parse_args()
    seed_all(args.seed)
    device = get_device()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # --- student backbone (official EMOv2 wrapper) ---
    student = StudentBackbone(variant=args.variant, head_hidden=args.head_hidden).to(device)

    # --- data loaders tied to student preprocessing ---
    train_ld, val_ld, img_size, mean, std = make_loaders(args, student)

    # --- teacher ---
    teacher = None
    if args.use_teacher:
        if args.teacher_kind == "v25":
            from teacher.v25_predictor import V25Teacher
            teacher = V25Teacher(device=device)
        else:
            from teacher.siglip_laion import SigLipTeacher
            teacher = SigLipTeacher(
                model_name=args.teacher_model,
                head_ckpt=args.teacher_head_ckpt,
                device=device
            )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    # --- optim & loss ---
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    mse = nn.MSELoss()

    # --- resume / early-stop state ---
    best_srcc = -1.0
    epochs_no_improve = 0
    start_epoch = 1

    # try to resume (expects a checkpoint saved with save_ckpt)
    if getattr(args, "resume", None):
        if os.path.isfile(args.resume):
            ck = load_ckpt(args.resume, model=student, optim=opt, map_location="cpu")
            extra = ck.get("extra", {})
            metrics_prev = extra.get("metrics", {})
            # prefer Teacher SRCC (KD) else fall back to GT SRCC if present
            best_srcc = metrics_prev.get("Teacher", {}).get("SRCC",
                        metrics_prev.get("GT", {}).get("SRCC", best_srcc))
            if best_srcc is None:
                best_srcc = -1.0
            start_epoch = int(extra.get("epoch", 0)) + 1
            print(f"[resume] {args.resume} | start_epoch={start_epoch} | best_SRCC={best_srcc:.4f}")
        else:
            print(f"[resume] file not found: {args.resume}")

    # --- training loop ---
    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        meter = AvgMeter()
        pbar = tqdm(train_ld, desc=f"Epoch {epoch}/{args.epochs}")
        for xb, yb, _ in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)  # GT in [0,1] (may be unused)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                yh = student(xb)                          # [B]
                sup_loss = mse(yh, yb) * args.lambda_supervised
                kd_loss = 0.0
                if teacher is not None:
                    tb = teacher(denorm_to_01(xb, mean, std))  # [B], teacher expects [0,1]
                    kd_loss = mse(yh, tb) * args.lambda_distill
                loss = sup_loss + kd_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            meter.update(loss.item(), xb.size(0))
            pbar.set_postfix(loss=f"{meter.avg:.4f}")

        # --- validation ---
        metrics = evaluate_loop(
            student, val_ld, device,
            teacher=teacher if args.use_teacher else None,
            mean=mean, std=std
        )
        print(f"[VAL] {json.dumps(metrics, indent=2)}")

        # 1) overwrite a rolling snapshot each epoch
        with open(os.path.join(args.out_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"epoch": epoch, **metrics}, f, indent=2)

        # 2) append a history line (one per epoch)
        with open(os.path.join(args.out_dir, "events.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": "val", "epoch": epoch, **metrics}) + "\n")

        # 3) (optional) keep per-epoch copies for auditing
        with open(os.path.join(args.out_dir, f"val_metrics_epoch_{epoch:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # choose SRCC to track (KD first, else GT)
        srcc_now = metrics.get("Teacher", {}).get("SRCC",
                   metrics.get("GT", {}).get("SRCC", -1.0))

        # always save "last"
        save_ckpt(
            os.path.join(args.out_dir, "ckpt_last.pt"),
            student, opt, epoch,
            extra={"metrics": metrics, "epoch": epoch, "variant": args.variant}
        )

        # best + early stopping
        if srcc_now > best_srcc + 1e-4:
            best_srcc = srcc_now
            epochs_no_improve = 0
            save_ckpt(
                os.path.join(args.out_dir, "ckpt_best.pt"),
                student, opt, epoch,
                extra={"metrics": metrics, "epoch": epoch, "variant": args.variant}
            )
            print(f"[save] new best SRCC={best_srcc:.4f}")

            # write a best snapshot for easy consumption
            with open(os.path.join(args.out_dir, "val_metrics_best.json"), "w", encoding="utf-8") as f:
                json.dump({"epoch": epoch, "SRCC": best_srcc, **metrics}, f, indent=2)

            # (optional): flag the best event in the JSONL history
            with open(os.path.join(args.out_dir, "events.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": "val_best", "epoch": epoch, "SRCC": best_srcc, **metrics}) + "\n")

        else:
            epochs_no_improve += 1
            print(f"[no improve] {epochs_no_improve}/{args.early_stop_patience}")

        if epochs_no_improve >= args.early_stop_patience:
            print("[early stop] stopping training due to no SRCC improvement.")
            break

        print(f"Done. Best SRCC: {best_srcc:.4f}")

if __name__ == "__main__":
    main()
