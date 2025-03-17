from torch.utils.data import DataLoader
from src_code.data_utils import FairFaceDataset
from src_code.model_utils import load_clip_model, inference_batch, get_topk_preds, encode_image_text


def main():

    model, preprocess = load_clip_model()

    # Prompts for the bias evaluation
    # TODO: improve
    fairface_labels = ["Male", "Female"]
    text_prompts = [f"A photo of a {label.lower()} person" for label in fairface_labels]

    # Single image inference
    # single_image = "datasets/fairface/train/images/1.jpg"
    # logits_per_image, _ = encode_image_text(model, preprocess, single_image, text_prompts)
    # topk_values, topk_indices = get_topk_preds(logits_per_image, k=2)
    # print(f"Image: {single_image}")
    # print("Top-K probabilities:", topk_values)
    # print("Top-K indices:", topk_indices)

    # Batch inference
    dataset = FairFaceDataset(image_dir="datasets/fairface/train/images", transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Iterate over a batch of images
    for images, image_paths in dataloader:
        logits_per_image, _ = inference_batch(model, images, text_prompts)

        # TODO: adjust top-k? only two options anyway
        topk_values, topk_indices = get_topk_preds(logits_per_image, k=2)

        # Top-k predictions for images in batch
        for i, path in enumerate(image_paths):
            print(f"Image: {path}")
            print("Top-K probabilities:", topk_values[i])
            print("Top-K indices:", topk_indices[i])
        break


if __name__ == '__main__':
    main()
