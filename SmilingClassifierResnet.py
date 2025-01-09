import torch
import torch.nn as nn
import os
from torch.optim import Adam
from torchvision.models import resnet34


class SmilingClassifierResnet(nn.Module):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.learning_rate=learning_rate
        model = resnet34(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.model = model
        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        return self.model(x)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
