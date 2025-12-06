import os, glob, pandas as pd, torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

def _read_merge_csv(root: str, split: str):
    """
    Expects TAD66K/labels/merge/train.csv and test.csv.
    Columns can be flexible; we try common names:
      image|img|filename|file
      label|score|mean|rating
    """
    mdir = os.path.join(root, "labels", "merge")
    csv_path = os.path.join(mdir, f"{'train' if split in ('train','val') else 'test'}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing merged CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    # find columns
    cols = {c.lower(): c for c in df.columns}
    imgc = next((cols[k] for k in ["image","img","filename","file","imagename","imgname"] if k in cols), None)
    scc  = next((cols[k] for k in ["label_mean","score","mean","rating","label"] if k in cols), None)
    themec = cols.get("theme", None)

    if not imgc:
        raise ValueError(f"CSV {csv_path} must contain an image filename column.")
    if scc is None:
        # KD-only mode: allow missing scores (fill with 0.0)
        df["_score"] = 0.0
        scc = "_score"

    df["_image"] = df[imgc].astype(str)
    df["_score"] = df[scc].astype(float, errors="ignore") if scc in df else 0.0
    # Map scores to [0,1] if they look like 1..5
    if df["_score"].max() > 1.5:
        df["_score"] = (df["_score"] - 1.0) / 4.0
    if themec:
        df["_theme"] = df[themec].astype(str)
    else:
        df["_theme"] = ""

    return df[["_image","_score","_theme"]].reset_index(drop=True)

class TAD66KDataset(Dataset):
    """
    Minimal TAD66K loader (train/val/test).
    - Images under TAD66K/images/
    - Labels from labels/merge/train.csv, test.csv
    - If labels absent/unused, still returns a float (0.0) for KD-only training.
    - Transforms are attached as self.tx and can be overridden by the trainer (same as AADB).
    """
    def __init__(self, root, split="train", resize=224, center_crop=False,
                 split_ratio=(0.8, 0.1, 0.1), seed=42, skip_missing=True):
        super().__init__()
        self.root = root
        self.img_dir = os.path.join(root, "images")

        df = _read_merge_csv(root, split)
        # Shuffle + split (train/val from train.csv)
        g = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n  = len(g)
        ntr = int(n*split_ratio[0]); nva = int(n*split_ratio[1])
        spans = {"train": (0, ntr), "val": (ntr, ntr+nva), "test": (0, n)} if split=="test" else \
                {"train": (0, ntr), "val": (ntr, ntr+nva)}
        s, e = spans[split]
        g = g.iloc[s:e].reset_index(drop=True)

        # Resolve image paths (exact name match; also try common extensions)
        items, miss = [], 0
        for name, score, theme in zip(g["_image"], g["_score"], g["_theme"]):
            base = os.path.normpath(os.path.join(self.img_dir, name))
            cand = []
            if os.path.splitext(base)[1]:
                cand.append(base)
            else:
                cand += [base + ext for ext in (".jpg", ".jpeg", ".png")]
            found = next((p for p in cand if os.path.exists(p)), None)
            if not found:
                miss += 1
                if skip_missing:
                    continue
                else:
                    items.append((None, score, name))
            else:
                items.append((found, score, os.path.basename(found)))
        if miss and skip_missing:
            print(f"[TAD66K:{split}] skipped {miss} missing files (kept {len(items)}) in {self.img_dir}")

        self.items = items

        # Default transforms (ImageNet stats); trainer may override (like AADB).
        aug = []
        if split == "train":
            aug += [transforms.RandomResizedCrop(resize, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip()]
        else:
            aug += [transforms.Resize(int(resize/0.875)),
                    transforms.CenterCrop(resize if center_crop else resize)]
        aug += [transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
        self.tx = transforms.Compose(aug)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        fp, score, name = self.items[idx]
        if fp is None:
            raise FileNotFoundError(f"Unresolved image path for {name}")
        img = Image.open(fp).convert("RGB")
        x = self.tx(img)
        y = torch.tensor(float(score), dtype=torch.float32)
        return x, y, name
