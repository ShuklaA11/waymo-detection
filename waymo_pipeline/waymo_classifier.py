"""ResNet18-based binary classifier: waymo vs not_waymo."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class WaymoClassifier:
    """Binary classifier for detecting Waymo vehicles from cropped images."""

    def __init__(self, weights_path: str, device: str = "mps"):
        self.device = device

        # Build model: ResNet18 with 2-class output
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

        # Load trained weights
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Classifier weights not found: {weights_path}\n"
                f"Train the classifier first:\n"
                f"  python training/train_classifier.py --data training/dataset/ --output {weights_path}"
            )

        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(device)

        # Preprocessing: match ImageNet normalization
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict(self, crop: np.ndarray) -> float:
        """
        Classify a cropped vehicle image.

        Args:
            crop: BGR numpy array (from OpenCV) of the cropped vehicle region.

        Returns:
            Probability of the vehicle being a Waymo (0.0 to 1.0).
        """
        if crop is None or crop.size == 0:
            return 0.0

        # Convert BGR to RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Transform and predict
        img_tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1)

        # Index 1 = waymo class
        return float(probs[0, 1].item())


class DummyClassifier:
    """
    Placeholder classifier that marks all vehicles as Waymo.
    Used for pipeline testing before the real classifier is trained.
    """

    def predict(self, crop: np.ndarray) -> float:
        return 1.0
