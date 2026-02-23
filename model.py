import torch
import torch.nn as nn
import torchvision.models as models

class SaccharumGAPNet(nn.Module):
    def __init__(self, backbone_name='resnet50', num_classes=5, pretrained=True):
        super(SaccharumGAPNet, self).__init__()
        
        # 1. Load Pretrained Backbone
        if backbone_name == 'resnet50':
            base_model = models.resnet50(weights='DEFAULT' if pretrained else None)
            # Remove the last FC layer
            self.backbone = nn.Sequential(*list(base_model.children())[:-2])
            self.in_features = 2048
        elif backbone_name == 'mobilenet_v3':
            base_model = models.mobilenet_v3_small(weights='DEFAULT' if pretrained else None)
            self.backbone = base_model.features
            self.in_features = 576
        else:
            raise ValueError("Backbone not supported. Choose 'resnet50' or 'mobilenet_v3'.")

        # 2. Global Average Pooling Layer
        # This reduces (Batch, Channels, H, W) -> (Batch, Channels, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 3. Lightweight Classifier Head
        # No hidden dense layers to minimize overfitting
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),  # Extra protection against overfitting
            nn.Linear(self.in_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

def get_model(name='resnet50', classes=5):
    model = SaccharumGAPNet(backbone_name=name, num_classes=classes)
    return model

if __name__ == "__main__":
    # Quick architecture test
    test_input = torch.randn(1, 3, 224, 224)
    model = get_model('resnet50')
    output = model(test_input)
    print(f"Model initialized. Output shape: {output.shape}") # Should be [1, 5]
