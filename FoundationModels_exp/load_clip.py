#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 18:16:01 2025

@author: MYQUEEN
"""
import clip
import torch
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_clip_features(images):
    features = []
    with torch.no_grad():
        for img in images:
            image_tensor = preprocess(img).unsqueeze(0).to(device)
            feature = model.encode_image(image_tensor)
            features.append(feature.squeeze().cpu().numpy())
    return np.array(features)
