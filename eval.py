import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import shutil

from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


# ------------------------------------------------
# Clean unwanted checkpoint folders
# ------------------------------------------------
def clean_checkpoints(folder_path):
    data_path = pathlib.Path(folder_path)

    for folder in data_path.rglob(".ipynb_checkpoints"):
        if folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)


# ------------------------------------------------
# Evaluation Function
# ------------------------------------------------
def evaluate_model(model_path, data_dir):

    # 🔥 Clean junk folders first
    clean_checkpoints(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------------------------------
    # 1️⃣ Transform (same as training)
    # ------------------------------------------------
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # ------------------------------------------------
    # 2️⃣ Load Dataset
    # ------------------------------------------------
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    classes = dataset.classes
    print("Class order:", classes)

    # ------------------------------------------------
    # 3️⃣ Load Model
    # ------------------------------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ------------------------------------------------
    # 4️⃣ Run Evaluation
    # ------------------------------------------------
    all_preds = []
    all_labels = []

    print("\nRunning Evaluation...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ------------------------------------------------
    # 5️⃣ Classification Report
    # ------------------------------------------------
    print("\n📊 Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # ------------------------------------------------
    # 6️⃣ Confusion Matrix
    # ------------------------------------------------
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=classes,
                yticklabels=classes)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Sugarcane Disease Confusion Matrix")

    os.makedirs("evaluation", exist_ok=True)
    plt.savefig("evaluation/confusion_matrix.png")
    plt.show()


# ------------------------------------------------
# RUN
# ------------------------------------------------
if __name__ == "__main__":

    evaluate_model(
        model_path="models/best_model.pth",
        data_dir="data/raw/test"   # ⚠️ Your new external dataset folder
    )
