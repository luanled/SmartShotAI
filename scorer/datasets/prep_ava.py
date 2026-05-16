import os
import json
from tqdm import tqdm


def clean_ava_labels():
    ava_root = r"..\..\data\AVA_dataset"
    input_txt = os.path.join(ava_root, "AVA.txt")
    output_json = os.path.join(ava_root, "ava_labels.json")
    img_dir = os.path.join(ava_root, "images")

    cleaned_data = []

    print("read AVA.txt...")
    with open(input_txt, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        parts = line.strip().split()
        if len(parts) < 12:
            continue

        img_id = parts[1]
        # vote[1-10] to int
        votes = [int(v) for v in parts[2:12]]

        # ensure image exist after extraction
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        if os.path.exists(img_path):
            cleaned_data.append({
                "img_id": img_id,
                "votes": votes
            })

    # save json
    with open(output_json, 'w') as f:
        json.dump(cleaned_data, f)

    print(f"\n Finish clean! Total images: {len(cleaned_data)}")
    print(f"Cleaned data saved to: {output_json}")


if __name__ == "__main__":
    clean_ava_labels()