import json
import os

from monai.data import PersistentDataset
import torch

from dinov3.data import InferenceAugmentation3d

def make_inference_dataset(dataset_json, dataset_path, patch_size):
    with open(dataset_json, "r") as f:
        dataset = json.load(f)

    for subject in dataset:
        full_path = os.path.join(dataset_path, subject["image"])
        subject["id"] = subject["image"] # track filepath
        subject["image"] = full_path

    transform = InferenceAugmentation3d(
        spacing=1,
        patch_size=16,
    )
    dataset = PersistentDataset(
        dataset,
        cache_dir = None,
        transform=transform
    )

    return dataset

def make_inference_dataloader(dataset, num_workers):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 1,
        num_workers = num_workers,
        pin_memory=True,
        shuffle=False
    )
    return data_loader # not sure if this is optimal, batch size fixed to 1 to avoid padding all images lol
