import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor
from PIL import Image
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

# ==========================================
# Dataset
# ==========================================
class AVADataset(Dataset):
    def __init__(self, csv_file, img_dir, processor):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.processor = processor

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data_frame.iloc[idx]['image'])

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0).contiguous()

        score = torch.tensor(self.data_frame.iloc[idx]['final_score'], dtype=torch.float32)
        return pixel_values, score


# ==========================================
# Model
# ==========================================
class SigLIPAestheticScorer(nn.Module):
    def __init__(self, model_name="google/siglip-base-patch16-224", unfreeze_last_block=False):
        super().__init__()

        base = AutoModel.from_pretrained(model_name)
        self.backbone = base.vision_model
        self.backbone.eval()

        if unfreeze_last_block:
            for name, param in self.backbone.named_parameters():
                if "encoder.layers.11" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        with torch.no_grad():
            outputs = self.backbone(x)

        raw = self.head(outputs.pooler_output).squeeze(-1)
        return raw * 9.0 + 1.0  # scale to 1–10

"""
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {frozen:,}")
"""

# ==========================================
# Evaluation
# ==========================================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    preds, labels_list = [], []
    val_loss = 0.0

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x = x.to(device).contiguous()
            y = y.to(device).float()

            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item()

            preds.extend(out.cpu().numpy())
            labels_list.extend(y.cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels_list)

    srcc, _ = spearmanr(preds, labels)
    plcc, _ = pearsonr(preds, labels)
    rmse = np.sqrt(np.mean((preds - labels)**2))

    return val_loss / len(dataloader), srcc, plcc, rmse


# ==========================================
# Training
# ==========================================
def train_ava(epochs, batch_size, lr, unfreeze_last_block=False):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Fine‑tuning SigLIP on AVA using {device}...")

    model_name = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    model = SigLIPAestheticScorer(model_name, unfreeze_last_block).to(device)
    
    # count_parameters(model)

    # Load TAD66K pretrained weights
    model.load_state_dict(torch.load("siglip_tad66k_best.pt", map_location=device))
    print("Loaded TAD66K pretrained SigLIP weights.")

    # AVA dataset
    train_ds = AVADataset("data/labels/ava/train.csv", "data/AVA/images", processor)
    val_ds   = AVADataset("data/labels/ava/test.csv",  "data/AVA/images", processor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Only train the head unless unfreezing last block
    optimizer = optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_srcc = -1
    epochs_without_improvement = 0
    patience = 3   # stop after 3 epochs with no improvement

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device).contiguous()
            y = y.to(device).float()

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type != "cpu")):
                out = model(x)
                loss = criterion(out, y)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, srcc, plcc, rmse = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Metrics -> SRCC: {srcc:.4f} | PLCC: {plcc:.4f} | RMSE: {rmse:.4f}")

        # Early stopping logic
        if srcc > best_srcc:
            best_srcc = srcc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "siglip_ava_finetuned.pt")
            print(">>> Best model saved!")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

            if epochs_without_improvement >= patience:
                print(">>> Early stopping triggered.")
                break

# ==========================================
# Run
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    train_ava(
        epochs=10,
        batch_size=32,
        lr=1e-3,
        unfreeze_last_block=False
    )
