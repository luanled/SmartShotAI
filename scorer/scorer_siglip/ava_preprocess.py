import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
AVA_IMAGES_DIR = "/Users/xinghebai/Desktop/Master Project/Scorer/data/AVA/images"
AVA_CSV = "/Users/xinghebai/Desktop/Master Project/Scorer/data/AVA/ground_truth_dataset.csv"
OUTPUT_DIR = "./data/labels/ava"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_mean_from_votes(row, vote_cols):
    ratings = row[vote_cols].values.astype(float)
    total = ratings.sum()
    if total == 0:
        return None
    return sum((i+1) * ratings[i] for i in range(10)) / total

def load_ground_truth(csv_path):
    df = pd.read_csv(csv_path)
    print("Loaded CSV columns:", df.columns.tolist())

    # Detect vote columns
    vote_cols = [f"vote_{i}" for i in range(1, 11)]
    if all(col in df.columns for col in vote_cols):
        print("Detected vote_1 ... vote_10 histogram columns.")
        df["final_score"] = df.apply(lambda row: compute_mean_from_votes(row, vote_cols), axis=1)
    else:
        raise ValueError("CSV missing vote_1 ... vote_10 columns.")

    # Build filename
    df["image"] = df["image_num"].astype(str) + ".jpg"

    # Check image existence
    df["exists"] = df["image"].apply(lambda x: os.path.exists(os.path.join(AVA_IMAGES_DIR, x)))
    df = df[df["exists"]]

    print(f"Valid images found: {len(df)}")

    return df[["image", "final_score"]]

def main():
    df = load_ground_truth(AVA_CSV)

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print("Saved:")
    print(" -", os.path.join(OUTPUT_DIR, "train.csv"))
    print(" -", os.path.join(OUTPUT_DIR, "test.csv"))

if __name__ == "__main__":
    main()
