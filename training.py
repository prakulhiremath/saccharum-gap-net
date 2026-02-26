import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
print(torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

def prepare_data(data_dir, batch_size):

    print("Loading dataset from:", data_dir)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load dataset
    full_dataset = datasets.ImageFolder(root=data_dir)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply transforms separately
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Classes:", full_dataset.classes)
    print("Total Images:", len(full_dataset))
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    return train_loader, val_loader, full_dataset.classes

    from data import prepare_data

    train_loader, val_loader, classes = prepare_data("data/raw/Sugarcane")

   import os

  data_dir = "data/raw/Sugarcane"  # replace with your dataset path

  for class_name in os.listdir(data_dir):
     class_path = os.path.join(data_dir, class_name)
      if os.path.isdir(class_path):
        images = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"{class_name} → {len(images)} images")

    def get_model(num_classes):

    model = models.resnet18(pretrained=True)

    # 🔥 Unfreeze entire network
    for param in model.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

   def run_training(epochs=25, lr=0.0001, batch_size=16):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    train_loader, val_loader, classes = prepare_data(
        "data/raw/Sugarcane",
        batch_size
    )

    model = get_model(len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        # ===== TRAINING =====
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader)

        for images, labels in train_bar:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_bar.set_description(
                f"Loss: {running_loss/len(train_loader):.4f} | Acc: {100*correct/total:.2f}%"
            )

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        preds = []
        true_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader):

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                preds.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_model.pth")
            print("⭐ Best model saved!")

    # ===== PLOTS =====

    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title("Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.show()

    plt.figure()
    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(["Train", "Validation"])
    plt.show()

    # ===== CONFUSION MATRIX =====
    cm = confusion_matrix(true_labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)

    plt.figure(figsize=(8, 8))
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


  run_training(epochs=25, lr=0.0001, batch_size=16)

  import os
  print(os.path.exists("models/best_model.pth"))
