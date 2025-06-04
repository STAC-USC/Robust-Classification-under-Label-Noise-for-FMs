#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 18:14:37 2025

@author: MYQUEEN
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Simulated comparison data for 4 foundation models on a downstream dataset
models = ["CLIP-ViT", "DINOv2", "SAM", "ResNet50"]
coverage_score = [0.89, 0.82, 0.68, 0.73]            # % of test points inside polytope
class_separability = [0.91, 0.88, 0.65, 0.70]        # 1 - (average overlap ratio between class polytopes)
nnk_sparsity = [4.1, 4.6, 6.9, 5.2]                  # lower is better (more sparse, more confident)
neighbor_consistency = [0.87, 0.84, 0.62, 0.75]      # Jaccard overlap of neighbors under augmentation

# Create dataframe
df = pd.DataFrame({
    "Model": models,
    "Polytope Coverage": coverage_score,
    "Class Separability": class_separability,
    "NNK Sparsity": nnk_sparsity,
    "Neighbor Consistency": neighbor_consistency
})

# Normalize for radar chart
df_norm = df.copy()
for col in df.columns[1:]:
    if "Sparsity" in col:  # inverse for radar (lower is better)
        df_norm[col] = 1 - (df[col] - min(df[col])) / (max(df[col]) - min(df[col]))
    else:
        df_norm[col] = (df[col] - min(df[col])) / (max(df[col]) - min(df[col]))

# Plot radar chart
labels = df.columns[1:]
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

for i, row in df_norm.iterrows():
    values = row[1:].tolist()
    values += values[:1]
    ax.plot(angles, values, label=row["Model"], linewidth=2)
    ax.fill(angles, values, alpha=0.1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title("Zero-Shot Geometric Benchmarking of Foundation Models", fontsize=14)
ax.legend(loc='lower right', bbox_to_anchor=(1.2, 0.1))
plt.tight_layout()
plt.show()
