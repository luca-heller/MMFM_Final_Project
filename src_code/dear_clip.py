import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class AdditiveResidualLearner(nn.Module):
    def __init__(self, embedding_dim):
        super(AdditiveResidualLearner, self).__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        # TODO: activation?
        self.activation = nn.Identity()

    def forward(self, x):
        residual = self.linear(x)
        residual = self.activation(residual)
        return residual


class DEAR_CLIP(nn.Module):
    def __init__(self, clip_model, arl):
        super(DEAR_CLIP, self).__init__()
        self.clip_model = clip_model
        self.arl = arl

    def encode_image(self, image):
        image_embedding = self.clip_model.encode_image(image)
        # Compute additive residual
        residual = self.arl(image_embedding)
        # Subtract residual to get debiased representation
        debiased_embedding = image_embedding - residual
        # Normalize debiased embedding
        debiased_embedding = debiased_embedding / debiased_embedding.norm(dim=-1, keepdim=True)
        return debiased_embedding

    def encode_text(self, text):
        text_embedding = self.clip_model.encode_text(text)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        return text_embedding

    def forward(self, image, text):
        debiased_image = self.encode_image(image)
        text_embedding = self.encode_text(text)
        # Compute cosine similarity logits
        logits = debiased_image @ text_embedding.t()
        return logits


class ProtectedAttributeClassifier(nn.Module):
    def __init__(self, embedding_dim, race_classes=7, age_classes=4, gender_classes=2):
        super(ProtectedAttributeClassifier, self).__init__()
        self.shared_fc = nn.Linear(embedding_dim, 256)
        self.relu = nn.ReLU()

        # Race head
        self.race_fc1 = nn.Linear(256, 128)
        self.race_fc2 = nn.Linear(128, race_classes)

        # Age head
        self.age_fc1 = nn.Linear(256, 128)
        self.age_fc2 = nn.Linear(128, age_classes)

        # Gender head
        self.gender_fc1 = nn.Linear(256, 128)
        self.gender_fc2 = nn.Linear(128, gender_classes)

    def forward(self, x):
        shared = self.relu(self.shared_fc(x))
        # Race preds
        race_hidden = self.relu(self.race_fc1(shared))
        race_logits = self.race_fc2(race_hidden)
        # Age preds
        age_hidden = self.relu(self.age_fc1(shared))
        age_logits = self.age_fc2(age_hidden)
        # Gender preds
        gender_hidden = self.relu(self.gender_fc1(shared))
        gender_logits = self.gender_fc2(gender_hidden)
        return race_logits, age_logits, gender_logits
