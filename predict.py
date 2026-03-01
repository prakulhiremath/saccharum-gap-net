import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt


# ------------------------------------------------
# Load Model
# ------------------------------------------------
def load_resnet18_model(num_classes, model_path, device):

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()

    return model


# ------------------------------------------------
# Prediction Function
# ------------------------------------------------
def predict_sugarcane_disease(image_path, model, class_names, device):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # 🔥 Print all probabilities
    print("\nAll class probabilities:")
    for i, prob in enumerate(probabilities[0]):
        print(f"{class_names[i]}: {prob.item()*100:.2f}%")

    confidence_score = confidence.item() * 100

    if confidence_score < 60:
        result_class = "Uncertain - Please retake image"
    else:
        result_class = class_names[predicted.item()]

    print("\nClass index mapping:")
    for i, c in enumerate(class_names):
        print(i, "→", c)

    print(f"\n🌿 Prediction: {result_class}")
    print(f"📊 Confidence: {confidence_score:.2f}%")

    plt.imshow(image)
    plt.title(f"{result_class} ({confidence_score:.1f}%)")
    plt.axis("off")
    plt.show()

    return result_class, confidence_score


# ------------------------------------------------
# RUN
# ------------------------------------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_dir = "data/raw/Sugarcane"

    dataset = datasets.ImageFolder(root=data_dir)
    classes = dataset.classes

    print("Loaded class order:", classes)

    model = load_resnet18_model(
        num_classes=len(classes),
        model_path="models/best_model.pth",
        device=device
    )

    predict_sugarcane_disease(
        image_path="hel.jpg",
        model=model,
        class_names=classes,
        device=device
    )
