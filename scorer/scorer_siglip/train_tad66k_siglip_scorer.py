import os
# Force Hugging Face to use a single thread to prevent the macOS mutex crash
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os
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
# 1. Dataset Definition
# ==========================================
class TAD66KDataset(Dataset):
    """
    Custom Dataset for TAD66K. 
    Reads 'train.csv' or 'test.csv' with 'image' and 'score' columns.
    """
    def __init__(self, csv_file, img_dir, processor):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.processor = processor

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Your CSV uses 'image' for the filename and 'score' for the GT
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx]['image'])
        image = Image.open(img_name).convert('RGB')
        
        # SigLIP processor handles 224px resize, center-crop, and normalization
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        pixel_values = pixel_values.contiguous()
        
        # Raw 1-10 Ground Truth score
        score = torch.tensor(self.data_frame.iloc[idx]['score'], dtype=torch.float32)
        
        return pixel_values, score

# ==========================================
# 2. Model Architecture
# ==========================================
class SigLIPAestheticScorer(nn.Module):
    def __init__(self, model_name="google/siglip-base-patch16-224"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name).vision_model
        
        # inside the Sequential block, which MPS prefers.
        self.head = nn.Sequential(
            nn.Linear(768, 256),
             nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )


    def forward(self, x):
        with torch.no_grad():
            outputs = self.backbone(x)
        return self.head(outputs.pooler_output).squeeze(-1)



# ==========================================
# 3. Training & Evaluation Functions
# ==========================================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device, memory_format=torch.contiguous_format).contiguous()
            labels = labels.to(device).float().contiguous()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate Metrics: SRCC, PLCC, RMSE
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    
    srcc, _ = spearmanr(preds, labels)
    plcc, _ = pearsonr(preds, labels)
    rmse = np.sqrt(np.mean((preds - labels)**2))

    return val_loss / len(dataloader), srcc, plcc, rmse

def train(epochs, batch_size, learning_rate):

    # Enable Apple Silicon GPU (MPS) or fallback to CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps") 
    else:
        device = torch.device("cpu")
        
    print(f"Training on {device}...")

    # Initialize Processor and Model
    model_name = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    model = SigLIPAestheticScorer(model_name).to(device)
    if device.type == "cuda": 
        model = model.to(memory_format=torch.channels_last)
    
    for p in model.backbone.parameters():
        p.requires_grad = False



    # Note: Ensure 'data/TAD66K/' points to the folder containing your actual .jpg images
    train_dataset = TAD66KDataset(csv_file='data/labels/merge/train.csv', img_dir='data/TAD66K', processor=processor)
    val_dataset = TAD66KDataset(csv_file='data/labels/merge/test.csv', img_dir='data/TAD66K', processor=processor)

    # Multi-processing disabled (num_workers=0) to prevent macOS mutex crashes
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = optim.AdamW(
        model.head.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    criterion = nn.MSELoss()

    best_srcc = -1.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs = inputs.to(device, memory_format=torch.contiguous_format).contiguous()
            labels = labels.to(device).float().reshape(-1).contiguous()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Ensure both are exactly the same shape and type
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        
        # Validation
        val_loss, srcc, plcc, rmse = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Metrics -> SRCC: {srcc:.4f} | PLCC: {plcc:.4f} | RMSE: {rmse:.4f}")

        # Save Best Model
        if srcc > best_srcc:
            best_srcc = srcc
            torch.save(model.state_dict(), "siglip_tad66k_best.pt")
            print(">>> Best model saved!")

# ==========================================
# 4. Entry Point
# ==========================================
if __name__ == "__main__":
    # Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-5 # Keep low for fine-tuning large vision models
    
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    train(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)