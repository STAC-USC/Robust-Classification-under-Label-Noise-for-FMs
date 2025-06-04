#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 19:33:06 2025

@author: MYQUEEN
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv("results.csv")

# Normalize metrics for radar plot (higher = better for all)
df_norm = df.copy()
df_norm["Centroid Distance"] = df_norm["Centroid Distance"].max() - df_norm["Centroid Distance"]  # invert
df_norm["Model"] = df["Model"]

# Radar plot
labels = ["kNN Accuracy", "Centroid Distance", "Silhouette Score"]
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
for i, row in df_norm.iterrows():
    values = row[labels].tolist()
    values += values[:1]
    ax.plot(angles, values, label=row["Model"], linewidth=2)
    ax.fill(angles, values, alpha=0.1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_title("Foundation Model Comparison (Radar Plot)")
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

# Bar plot
df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
plt.figure(figsize=(8, 5))
for metric in labels:
    subset = df_melted[df_melted["Metric"] == metric]
    plt.bar(subset["Model"], subset["Score"], label=metric)
    plt.xticks(rotation=45)
plt.title("Foundation Model Metric Comparison")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.show()