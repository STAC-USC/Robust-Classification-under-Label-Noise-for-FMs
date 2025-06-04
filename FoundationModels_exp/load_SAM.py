#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 18:18:35 2025

@author: MYQUEEN
"""

from segment_anything import sam_model_registry
import torchvision.transforms as T
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth").to(device)
transform = T.Compose([
    T.Resize(1024),
    T.CenterCrop(1024),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def extract_sam_features(images):
    features = []
    for img in images:
        image_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = sam.image_encoder(image_tensor)
        features.append(out.mean(dim=(-1, -2)).squeeze().cpu().numpy())  # global pooled
    return np.array(features)
