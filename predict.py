import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
# Assuming your previous scripts are accessible
# from models.model import get_model
# from data import get_saccharum_transforms

def predict_sugarcane_disease(image_path, model_path, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    # model = get_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Preprocess Image
    # Using the 'test' version of transforms (no augmentation)
    transform = get_saccharum_transforms(train=False)
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension

    # 3. Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        conf, pred = torch.max(probabilities, 1)

    # 4. Result Mapping
    result_class = class_names[pred.item()]
    confidence_score = conf.item() * 100

    print(f"🌿 Prediction: {result_class}")
    print(f"📊 Confidence: {confidence_score:.2f}%")

    # 5. Visual Output
    plt.imshow(image)
    plt.title(f"Predicted: {result_class} ({confidence_score:.1f}%)")
    plt.axis('off')
    plt.show()

    return result_class, confidence_score

if __name__ == "__main__":
    # Example Usage:
    # classes = ['Healthy', 'Mosaic', 'Redrot', 'Rust', 'Yellow']
    # predict_sugarcane_disease('data/test_image.jpg', 'models/checkpoints/best_saccharum_gap.pth', classes)
    pass
