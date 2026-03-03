import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image


# -----------------------------
# GradCAM Class
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, class_idx=None):

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam / torch.max(cam)

        return cam.detach().cpu().numpy()


# -----------------------------
# Load Model
# -----------------------------
def load_model(model_path, num_classes, device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# -----------------------------
# Run GradCAM
# -----------------------------
def run_gradcam(image_path, model_path, class_names):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, len(class_names), device)

    target_layer = model.layer4  # 🔥 Important
    gradcam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    cam = gradcam.generate_cam(input_tensor)

    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    original = cv2.resize(np.array(image), (224, 224))
    superimposed = heatmap * 0.4 + original

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(heatmap)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(np.uint8(superimposed))
    plt.axis("off")

    plt.show()


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":

    class_names = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

    run_gradcam(
        image_path="hell.jpg",  # change to any image
        model_path="models/best_model.pth",
        class_names=class_names
    )
