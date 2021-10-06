import torch
from torch import nn
import torch
import torchvision
from common import normalize, resize

class ViewRegressor(nn.Module):
    def __init__(self, backbone='resnext101_32x8d', out_dim=4):
        super().__init__()

        if backbone == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
        elif backbone == 'resnext101_32x8d':
            model = torchvision.models.resnext101_32x8d(pretrained=True)
        else:
            raise Exception('Model is not supported!')
    
        num_feat = model.fc.in_features
        model.fc = nn.Linear(num_feat, out_dim)
        self.model = model

    def forward(self, x, transform=False):
        if transform:
            x = normalize(resize(x))
        return self.model(x)