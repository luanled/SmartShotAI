import os, re
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from urllib.parse import urlparse

_URL_COLS = [f"Input.image_url{i}" for i in range(1, 11)]
_SCORE_COLS = [f"Answer.overallScore{i}" for i in range(1, 11)]

def _explode_allinall(df):
    rows = []
    for _, r in df.iterrows():
        for i in range(10):
            url = r.get(_URL_COLS[i], None)
            sc  = r.get(_SCORE_COLS[i], None)
            if pd.isna(url) or pd.isna(sc):
                continue
            try:
                score = float(sc)
            except Exception:
                continue
            score = (score - 1.0) / 4.0 if score > 1.5 else score  # 1..5 → [0,1]
            rows.append((str(url).strip(), score))
    return pd.DataFrame(rows, columns=["url", "score"])

def _url_candidates(url: str):
    """Map AADB Flickr URL to possible local filenames."""
    try:
        u = urlparse(url)
        host = u.netloc  # e.g., farm1.staticflickr.com
        path_parts = [p for p in u.path.split("/") if p]
        basename = path_parts[-1]                         # 19452343093_8ee7e5e375_b.jpg
        server = path_parts[0] if len(path_parts) >= 2 else None  # 255
        m = re.match(r"farm(\d+)\.staticflickr\.com", host)
        farm = f"farm{m.group(1)}" if m else None

        cands = [basename]
        if server: cands.append(f"{server}_{basename}")
        if farm:   cands.append(f"{farm}_{basename}")
        if farm and server: cands.append(f"{farm}_{server}_{basename}")
        # also try .jpeg vs .jpg swap (some zips differ)
        more = []
        for c in cands:
            if c.lower().endswith(".jpg"):
                more.append(c[:-4] + ".jpeg")
            if c.lower().endswith(".jpeg"):
                more.append(c[:-5] + ".jpg")
        return list(dict.fromkeys(cands + more))  # unique, keep order
    except Exception:
        return [os.path.basename(url)]

class AADBDataset(Dataset):
    """
    Loads AADB from AllinAll.csv, resolves local filenames in the chosen img_subdir.
    Only uses images under that subdir (no other folders).
    """
    def __init__(self, root, split="train", csv_path=None, img_subdir="datasetImages_originalSize",
                 resize=224, center_crop=False, split_ratio=(0.8, 0.1, 0.1), seed=42, skip_missing=True):
        super().__init__()
        self.root = root
        self.base_dir = os.path.join(root, img_subdir)
        os.makedirs(self.base_dir, exist_ok=True)  # no-op if exists

        csv_path = csv_path or os.path.join(root, "AllinAll.csv")
        df = pd.read_csv(csv_path, low_memory=False)

        # Wide → long
        if set(_URL_COLS).issubset(df.columns) and set(_SCORE_COLS).issubset(df.columns):
            df_long = _explode_allinall(df)
        else:
            # Fallback simple CSV with columns: image, score/label_mean
            cols = {c.lower(): c for c in df.columns}
            imgc = next((cols[k] for k in ["image","img","filename","file","imagename","imgname"] if k in cols), None)
            scc  = next((cols[k] for k in ["label_mean","score","mean","rating","meanscore"] if k in cols), None)
            if not imgc or not scc:
                raise ValueError("CSV is not AADB AllinAll nor simple (image,label).")
            df_long = pd.DataFrame({"url": df[imgc].astype(str), "score": df[scc].astype(float)})
            if df_long["score"].max() > 1.5:
                df_long["score"] = (df_long["score"] - 1.0)/4.0

        # Shuffle + split
        g = df_long.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n  = len(g)
        ntr = int(n*split_ratio[0]); nva = int(n*split_ratio[1])
        spans = {"train": (0, ntr), "val": (ntr, ntr+nva), "test": (ntr+nva, n)}
        s, e = spans[split]
        g = g.iloc[s:e].reset_index(drop=True)

        # Resolve local files now (and optionally drop missing)
        items, miss = [], 0
        for url, score in zip(g["url"], g["score"]):
            found_path = None
            for cand in _url_candidates(url):
                p = os.path.normpath(os.path.join(self.base_dir, cand))
                if os.path.exists(p):
                    found_path = p
                    break
            if found_path is None:
                miss += 1
                if skip_missing:
                    continue
                else:
                    # keep unresolved; __getitem__ will raise
                    items.append((url, score, None))
            else:
                items.append((found_path, score, os.path.basename(found_path)))
        if miss and skip_missing:
            print(f"[AADB:{split}] skipped {miss} missing files (kept {len(items)}) in {self.base_dir}")

        self.items = items

        # Transforms (ImageNet stats; adjust if swap student backbone)
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

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fp, score, name = self.items[idx]
        if fp is None:
            raise FileNotFoundError("Unresolved image path")
        img = Image.open(fp).convert("RGB")
        x = self.tx(img)
        y = torch.tensor(score, dtype=torch.float32)
        return x, y, name
