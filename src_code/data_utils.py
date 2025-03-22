import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# Mappings for attributes
race_mapping = {
    "White": 0,
    "Black": 1,
    "Indian": 2,
    "East Asian": 3,
    "Southeast Asian": 4,
    "Middle Eastern": 5,
    "Latino_Hispanic": 6
}

age_mapping = {
    '0-2': 0,
    '3-9': 1,
    '10-19': 2,
    '20-29': 3,
    '30-39': 4,
    '40-49': 5,
    '50-59': 6,
    '60-69': 7,
    'more than 70': 8
}

gender_mapping = {
    "Male": 0,
    "Female": 1
}


class FairFaceDataset(Dataset):
    def __init__(self, image_dir, label_csv, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.labels_df = pd.read_csv(label_csv)
        self.label_dict = {
            os.path.basename(row["file"]).lower(): (row["race"], row["age"], row["gender"])
            for _, row in self.labels_df.iterrows()
        }
        all_image_paths = [os.path.join(image_dir, fname)
                           for fname in os.listdir(image_dir)
                           if fname.lower().endswith(".jpg")]
        self.image_paths = [p for p in all_image_paths if os.path.basename(p) in self.label_dict]
        print(f"Found {len(self.image_paths)} images with labels.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        filename = os.path.basename(image_path)
        race_label_str, age_label_str, gender_label_str = self.label_dict[filename]
        # Convert categorical labels to integers
        race_label = race_mapping[race_label_str]
        age_label = age_mapping[age_label_str]
        gender_label = gender_mapping[gender_label_str]
        labels = (race_label, age_label, gender_label)
        return image, labels
