import os
import torch
import pathlib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import shutil


# ============================================================
# Remove Jupyter checkpoint folders (VERY IMPORTANT)
# ============================================================

def clean_dataset_directory(data_dir):
    data_path = pathlib.Path(data_dir)

    for checkpoint_folder in data_path.rglob(".ipynb_checkpoints"):
        if checkpoint_folder.is_dir():
            shutil.rmtree(checkpoint_folder, ignore_errors=True)

    print("✅ Dataset directory cleaned")


# ============================================================
# Transform Pipeline
# ============================================================

def get_saccharum_transforms(img_size=224, train=True):

    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )

    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])


# ============================================================
# Dataset Preparation
# ============================================================

def prepare_data(data_dir, batch_size=32, split_ratio=0.8):

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset not found at {data_dir}")

    clean_dataset_directory(data_dir)

    # Full dataset (no transform yet)
    full_dataset = datasets.ImageFolder(root=data_dir)

    dataset_size = len(full_dataset)
    train_size = int(split_ratio * dataset_size)
    val_size = dataset_size - train_size

    # Random split (stable)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )

    # Assign transforms separately
    train_dataset.dataset.transform = get_saccharum_transforms(train=True)
    val_dataset.dataset.transform = get_saccharum_transforms(train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    print(f"✅ Data Ready: {train_size} training images, {val_size} validation images.")
    print(f"📋 Classes detected: {full_dataset.classes}")

    return train_loader, val_loader, full_dataset.classes


# ============================================================
# Test Run
# ============================================================

if __name__ == "__main__":
    try:
        train_loader, val_loader, classes = prepare_data("data/raw/Sugarcane")
    except Exception as e:
        print("Setup Note:", e)
