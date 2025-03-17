import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_model(model_name="ViT-B/32"):
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess


def get_topk_preds(logits, k=2):
    probs = logits.softmax(dim=-1)
    topk_values, topk_indices = torch.topk(probs, k, dim=-1)
    return topk_values, topk_indices


def encode_image_text(model, preprocess, image_path, text_list):
    # Encode image and list of text prompts using CLIP
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(text_list).to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
    return logits_per_image, logits_per_text


def inference_batch(model, images, text_list):
    text = clip.tokenize(text_list).to(device)
    images = images.to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = model(images, text)
    return logits_per_image, logits_per_text
