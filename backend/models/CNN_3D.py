import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


class CNN3D(nn.Module):
    """
    3D CNN based on torchvision r3d_18.

    Input : (batch, 3, 16, 112, 112)  — RGB, [0, 1]
    Output: (batch, 4) logits

    Class order: 0=normal, 1=trespassing, 2=loitering, 3=object_abandonment
    """

    class_names = ['normal', 'trespassing', 'loitering', 'object_abandonment']

    def __init__(self, num_classes: int = 4, dropout: float = 0.5, **kwargs):
        super().__init__()
        base = r3d_18(weights=None)
        in_features = base.fc.in_features  # 512

        # Match training config: Sequential(Dropout, Linear)
        base.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        # Bind r3d_18 submodules directly so state_dict keys match the checkpoint
        self.stem    = base.stem
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool
        self.fc      = base.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# Backward compatibility alias
CNN_3D = CNN3D
