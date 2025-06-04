#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 18:19:41 2025

@author: MYQUEEN
"""

import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as T
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = models.resnet50(pretrained=True).to(device)
resnet.fc = nn.Identity()  # Remove final classification head

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def extract_resnet_features(images):
    features = []
    with torch.no_grad():
        for img in images:
            image_tensor = transform(img).unsqueeze(0).to(device)
            feature = resnet(image_tensor)
            features.append(feature.squeeze().cpu().numpy())
    return np.array(features)
