#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 18:17:27 2025

@author: MYQUEEN
"""

import torch
import torchvision.transforms as T
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

def extract_dino_features(images):
    features = []
    for img in images:
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs).last_hidden_state[:, 0, :]  # CLS token
        features.append(output.squeeze().cpu().numpy())
    return np.array(features)
