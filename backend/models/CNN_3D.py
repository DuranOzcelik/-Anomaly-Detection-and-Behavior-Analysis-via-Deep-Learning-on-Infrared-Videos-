import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3D(nn.Module):
    """
    3D Convolutional Neural Network for video anomaly classification
    Predicts: 'Normal', 'Loitering', 'Trespass', 'Obj. Aband.'
    """

    def __init__(self, num_classes=4, input_channels=3, depth=10):
        super(CNN3D, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.depth = depth

        self.class_names = ['Normal', 'Loitering', 'Trespass', 'Obj. Aband.']

        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(1, 1, 1)
        )
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(1, 1, 1)
        )
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = nn.Conv3d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        self.bn4 = nn.BatchNorm3d(256)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # Adaptive average pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Forward pass
        x: (batch, channels, depth, height, width)
        returns: logits of shape (batch, num_classes)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        # Adaptive pooling
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x

    def predict(self, x):
        """
        Get class predictions with probabilities
        x: input tensor (batch, channels, depth, height, width)
        returns: dict with predictions and probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        return {
            'logits': logits,
            'probabilities': probabilities,
            'predictions': predictions,
            'class_names': [self.class_names[p.item()] for p in predictions],
            'confidence': probabilities.max(dim=1).values
        }

    def get_class_name(self, class_idx):
        """Get class name from index"""
        if 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return "Unknown"


class CNN_3D(CNN3D):
    """Alias for CNN3D"""
    pass
