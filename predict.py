import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


def load_resnet18_model(num_classes, model_path, device):

    # Create same model as training
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()

    return model


def predict_sugarcane_disease(image_path, model_path, class_names):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = load_resnet18_model(
        num_classes=len(class_names),
        model_path=model_path,
        device=device
    )

    # SAME normalization used during training
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    result_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100

    print(f"\n🌿 Prediction: {result_class}")
    print(f"📊 Confidence: {confidence_score:.2f}%")

    plt.imshow(image)
    plt.title(f"{result_class} ({confidence_score:.1f}%)")
    plt.axis("off")
    plt.show()

    return result_class, confidence_score


# ===== RUN TEST =====
if __name__ == "__main__":

    classes = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

    predict_sugarcane_disease(
        image_path="mo.jpg",
        model_path="models/best_model.pth",
        class_names=classes
    )
