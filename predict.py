import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from model import get_model
from data import get_saccharum_transforms


def predict_sugarcane_disease(image_path, model_path, class_names):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = get_model(name='resnet50', classes=len(class_names)).to(device)

    # Load weights (IMPORTANT)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    model.eval()

    # Preprocess image
    transform = get_saccharum_transforms(train=False)

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)

        conf, pred = torch.max(probabilities, 1)

    result_class = class_names[pred.item()]
    confidence_score = conf.item() * 100

    print(f"🌿 Prediction: {result_class}")
    print(f"📊 Confidence: {confidence_score:.2f}%")

    plt.imshow(image)
    plt.title(f"Predicted: {result_class} ({confidence_score:.1f}%)")
    plt.axis('off')
    plt.show()

    return result_class, confidence_score


classes = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

predict_sugarcane_disease(
    image_path="redrot.jpg",
    model_path="models/best_model.pth",
    class_names=classes
)
