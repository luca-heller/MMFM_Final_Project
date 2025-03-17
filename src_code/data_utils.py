import os
from PIL import Image
from torch.utils.data import Dataset


class FairFaceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        # transform applied on the image: CLIP preprocess function
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname)
                            for fname in os.listdir(image_dir)
                            if fname.lower().endswith((".jpg"))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path
