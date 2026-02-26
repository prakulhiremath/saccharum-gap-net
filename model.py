import torch
import torch.nn as nn
import torchvision.models as models

class SaccharumGAPNet(nn.Module):
    """
    Global Average Pooling CNN for Sugarcane Disease Classification
    Supports:
    - ResNet50
    - MobileNetV3 Small
    """

    def __init__(self, backbone_name='resnet50', num_classes=5, pretrained=True):
        super(SaccharumGAPNet, self).__init__()

        # -------------------------------
        # Backbone Selection
        # -------------------------------
        if backbone_name == 'resnet50':
            base_model = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )

            # Remove FC layer + pooling layer
            self.backbone = nn.Sequential(*list(base_model.children())[:-2])
            self.in_features = 2048

        elif backbone_name == 'mobilenet_v3':
            base_model = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            )

            self.backbone = base_model.features
            self.in_features = 576

        else:
            raise ValueError("Backbone must be 'resnet50' or 'mobilenet_v3'")

        # -------------------------------
        # Global Average Pooling
        # -------------------------------
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # -------------------------------
        # Classifier Head (Lightweight)
        # -------------------------------
        self.classifier = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# -------------------------------
# Model Loader Function
# -------------------------------
def get_model(name='resnet50', classes=5):
    return SaccharumGAPNet(
        backbone_name=name,
        num_classes=classes
    )


# -------------------------------
# Quick Architecture Test
# -------------------------------
if __name__ == "__main__":
    test_input = torch.randn(1, 3, 224, 224)
    model = get_model('resnet50')
    output = model(test_input)

    print("Model initialized successfully!")
    print("Output shape:", output.shape)
