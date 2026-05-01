import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor
from PIL import Image
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TAD66KDataset(Dataset):
    def __init__(self, csv_file, img_dir, processor):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.processor = processor

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx]['image'])
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        score = torch.tensor(self.data_frame.iloc[idx]['score'], dtype=torch.float32)
        return pixel_values, score

class SigLIPAestheticScorer(nn.Module):
    def __init__(self, model_name="google/siglip-base-patch16-224"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name).vision_model
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Flatten(0) 
        )

    def forward(self, x):
        outputs = self.backbone(x)
        return self.head(outputs.pooler_output)

def evaluate_test_set():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    model_name = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    model = SigLIPAestheticScorer(model_name).to(device)

    # Load your saved weights!
    weight_path = "siglip_tad66k_best.pt"
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval() 

    # --- UPDATE THESE PATHS IF NEEDED ---
    test_csv = 'data/labels/merge/test.csv'
    img_dir = 'data/TAD66K/'
    
    test_dataset = TAD66KDataset(csv_file=test_csv, img_dir=img_dir, processor=processor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device).float().view(-1)
            outputs = model(images).view(-1)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    
    srcc, _ = spearmanr(preds, labels)
    plcc, _ = pearsonr(preds, labels)
    rmse = np.sqrt(np.mean((preds - labels)**2))

    print("\n" + "="*40)
    print("FINAL TEST SET PERFORMANCE")
    print("="*40)
    print(f"SRCC: {srcc:.4f}")
    print(f"PLCC: {plcc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_test_set()