import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3


class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        base_model = efficientnet_b3(weights=None)
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # ⚠️ EĞİTİMLE BİREBİR AYNI
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1536, 512),
            nn.LayerNorm(512),
            nn.ReLU6(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU6(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def load_model(weight_path, num_classes, device):
    model = CustomEfficientNet(num_classes)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)   # ✅ artık hata YOK
    model.to(device)
    model.eval()
    return model
