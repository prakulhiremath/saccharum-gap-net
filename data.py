import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_saccharum_transforms(img_size=224, train=True):
    """
    Custom transforms for sugarcane leaves.
    Includes field-specific augmentations like color jitter and rotation.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            # Simulates varying sunlight/soil conditions
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def prepare_data(data_dir, batch_size=32, split_ratio=0.8):
    """
    Loads the Sugarcane Leaf Disease Dataset and splits into Train/Val sets.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found at {data_dir}. "
                                "Please download the Mendeley dataset (DOI: 10.17632/9424skmnrk.1)")

    # Load full dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=get_saccharum_transforms(train=True))
    
    # Calculate split lengths
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update validation transforms (remove augmentations)
    val_dataset.dataset.transform = get_saccharum_transforms(train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"✅ Data Ready: {train_size} training images, {val_size} validation images.")
    print(f"📋 Classes detected: {full_dataset.classes}")
    
    return train_loader, val_loader, full_dataset.classes

if __name__ == "__main__":
    # Quick test run
    # Change 'data/raw' to your actual local path
    try:
        train, val, classes = prepare_data('data/raw')
    except Exception as e:
        print(f"Setup Note: {e}")
