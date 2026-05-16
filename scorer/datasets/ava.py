import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class AVADataset(Dataset):
    def __init__(self, root_dir, split="train", resize=224):
        self.img_dir = os.path.join(root_dir, "images")
        self.tx = None  # assigned later by build_transforms

        json_path = os.path.join(root_dir, "ava_labels.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Cannot find {json_path}，please run prep_ava.py first！")

        with open(json_path, 'r') as f:
            self.data = json.load(f)

        val_size = 20000
        if split == "train":
            self.data = self.data[:-val_size]
        else:
            self.data = self.data[-val_size:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_id = item["img_id"]
        votes = item["votes"]

        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        # if bad img
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # print warning and replace another image
            print(f"Warning: Failed to load image {img_path}. Error: {e}. Sampling a random image instead.")
            import random
            return self.__getitem__(random.randint(0, len(self.data) - 1))
        # --------------------------

        if self.tx is not None:
            image = self.tx(image)

        votes_tensor = torch.tensor(votes, dtype=torch.float32)
        p_true = votes_tensor / (votes_tensor.sum() + 1e-8)

        return image, p_true, img_id