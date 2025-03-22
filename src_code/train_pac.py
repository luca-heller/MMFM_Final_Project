import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from dear_clip import ProtectedAttributeClassifier
from model_utils import load_clip_model

device = "cuda" if torch.cuda.is_available() else "cpu"

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


# TODO: combine with class in data_utils
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


def train_pac(pac, clip_model, dataloader, criterion, optimizer, num_epochs=10, checkpoint_dir="checkpoints", checkpoint_interval=1):
    clip_model.eval()
    pac.train()
    os.makedirs(checkpoint_dir, exist_ok=True)
    epoch_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(dataloader):
            race_labels, age_labels, gender_labels = labels
            images = images.to(device)
            race_labels = torch.tensor(race_labels, dtype=torch.long, device=device)
            age_labels = torch.tensor(age_labels, dtype=torch.long, device=device)
            gender_labels = torch.tensor(gender_labels, dtype=torch.long, device=device)

            # Get debiased image embeddings using CLIP
            with torch.no_grad():
                image_embeddings = clip_model.encode_image(images)
                image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

            race_logits, age_logits, gender_logits = pac(image_embeddings)

            # Compute cross-entropy losses
            loss_race = criterion(race_logits, race_labels)
            loss_age = criterion(age_logits, age_labels)
            loss_gender = criterion(gender_logits, gender_labels)
            loss = loss_race + loss_age + loss_gender

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

        # Save checkpoints
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': pac.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    return pac, epoch_losses


def main():
    clip_model, preprocess = load_clip_model("ViT-B/32")
    embedding_dim = 512

    pac = ProtectedAttributeClassifier(embedding_dim, race_classes=7, age_classes=9, gender_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pac.parameters(), lr=5e-3)

    image_dir = "../datasets/fairface/train/images"
    label_csv = "../datasets/fairface/train/fairface_label_train.csv"
    dataset = FairFaceDataset(image_dir=image_dir, label_csv=label_csv, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    pac, epoch_losses = train_pac(pac, clip_model, dataloader, criterion, optimizer, num_epochs=10)

    # Freeze PAC parameters and save trained model
    for param in pac.parameters():
        param.requires_grad = False
    torch.save(pac.state_dict(), "trained_pac.pth")

    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
