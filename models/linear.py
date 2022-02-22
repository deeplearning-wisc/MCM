import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, feat_dim,  normalize = False, num_classes=100):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        self.normalize = normalize

    def forward(self, features):
        if self.normalize: 
            features =  F.normalize(features, dim=1)
        return self.fc(features)