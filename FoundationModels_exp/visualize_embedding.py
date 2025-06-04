#!/usr/bin/env python3
"""
visualize_embedding.py

Standalone script integrated with run_benchmark.py cache:
  - CLI flags: --dataset, --model, --n-per-class
  - Supports CIFAR10, STL10, and any MedMNIST dataset key (e.g. dermamnist)
  - Produces three figures in ./vis:
      1) Plain PCA scatter (2D)
      2) Overlay scatter with small thumbnails (2D)
      3) PCA-3 scatter (3D)

Usage:
  chmod +x visualize_embedding.py
  ./visualize_embedding.py --dataset cifar10 --model clip --n-per-class 25
  ./visualize_embedding.py --dataset dermamnist --model clip --n-per-class 25
"""
import os
import argparse
import random
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.linear_model import LogisticRegression
# dataset loading
import medmnist
from medmnist import INFO
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage


def load_dataset(name: str, n_per_class: int):
    name = name.lower()
    # MedMNIST
    if name in INFO:
        info = INFO[name]
        cls = getattr(medmnist, info['python_class'])
        tr_ds = cls(split='train', download=True)
        te_ds = cls(split='test',  download=True)
        tr_imgs = [Image.fromarray(img).convert('RGB') for img in tr_ds.imgs]
        te_imgs = [Image.fromarray(img).convert('RGB') for img in te_ds.imgs]
        tr_lbls = tr_ds.labels.flatten()
        te_lbls = te_ds.labels.flatten()
        print(f"[LOAD] MedMNIST ({name}): {len(tr_imgs)} train, {len(te_imgs)} test")
        if n_per_class > 0:
            def sub(imgs, lbls):
                buckets, out_i, out_l = {}, [], []
                for im, lb in zip(imgs, lbls):
                    buckets.setdefault(lb, [])
                    if len(buckets[lb]) < n_per_class:
                        buckets[lb].append(im)
                for lb in sorted(buckets):
                    out_i.extend(buckets[lb])
                    out_l.extend([lb]*len(buckets[lb]))
                return out_i, np.array(out_l)
            tr_imgs, tr_lbls = sub(tr_imgs, tr_lbls)
            te_imgs, te_lbls = sub(te_imgs, te_lbls)
            print(f"[SUBSAMPLE] MedMNIST: {len(tr_imgs)} train, {len(te_imgs)} test")
        return tr_imgs, tr_lbls, te_imgs, te_lbls

    # CIFAR10 / STL10
    tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    if name == 'cifar10':
        tr = datasets.CIFAR10('./data', train=True, download=True, transform=tfm)
        te = datasets.CIFAR10('./data', train=False,download=True, transform=tfm)
    elif name == 'stl10':
        tr = datasets.STL10('./data', split='train',download=True,transform=tfm)
        te = datasets.STL10('./data', split='test', download=True,transform=tfm)
    else:
        raise ValueError(f"Unknown dataset {name}")
    def sub(ds):
        buckets, imgs, lbls = {}, [], []
        for img, lbl in ds:
            buckets.setdefault(lbl, [])
            if len(buckets[lbl]) < n_per_class:
                buckets[lbl].append(ToPILImage()(img))
        for lb in sorted(buckets):
            imgs.extend(buckets[lb])
            lbls.extend([lb]*len(buckets[lb]))
        return imgs, np.array(lbls)
    tr_imgs, tr_lbls = sub(tr)
    te_imgs, te_lbls = sub(te)
    print(f"[LOAD] {name}: {len(tr_imgs)} train, {len(te_imgs)} test")
    return tr_imgs, tr_lbls, te_imgs, te_lbls


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(INFO.keys())+['cifar10','stl10'], required=True)
    p.add_argument('--model',   choices=['clip','dino'], required=True)
    p.add_argument('--n-per-class', type=int, required=True)
    args=p.parse_args()

    print(f"Loading dataset '{args.dataset}' with {args.n_per_class} per class...")
    tr_imgs, tr_lbls, te_imgs, te_lbls = load_dataset(args.dataset, args.n_per_class)
    Ntr, Nte = len(tr_imgs), len(te_imgs)

    print(f"Loading cached features for {args.model}...")
    cache_t = Path('cache')/f'cached_{args.dataset}_{args.model}_test_{Nte}.npy'
    cache_tr= Path('cache')/f'cached_{args.dataset}_{args.model}_train_{Ntr}.npy'
    if not cache_t.exists() or not cache_tr.exists():
        raise FileNotFoundError('Run run_benchmark.py first')
    Xte = np.load(cache_t); print(f"Loaded X_test {Xte.shape}")
    Xtr = np.load(cache_tr); print(f"Loaded X_train {Xtr.shape}")

    #print("Computing k-NN correctness mask...")
    #y_pred = KNeighborsClassifier(5).fit(Xtr, tr_lbls).predict(Xte)
    #mask = (y_pred==te_lbls).astype(int)
    #print(f"Mask: {mask.sum()} correct, {len(mask)-mask.sum()} misclassified")
    
    print("Computing log regression  correctness mask...")
    lr = LogisticRegression(max_iter=1000, random_state=0)
    lr.fit(Xtr, tr_lbls)
    y_pred = lr.predict(Xte)
    mask   = (y_pred == te_lbls).astype(int)
    print(f"Mask: {mask.sum()} correct, {len(mask)-mask.sum()} misclassified")

    print("Running PCA (2D)...")
    X2 = PCA(2, random_state=0).fit_transform(Xte)
    print("Sampling examples...")
    rng=random.Random(0)
    corr=[i for i,m in enumerate(mask) if m]
    mis =[i for i,m in enumerate(mask) if not m]
    sel_corr=rng.sample(corr,5); sel_mis=rng.sample(mis,5)

    os.makedirs('vis',exist_ok=True)
    # 1) Plain scatter
    print("Plotting plain 2D scatter...")
    plt.figure(figsize=(8,6))
    plt.scatter(X2[:,0],X2[:,1],c=mask,cmap='coolwarm',s=10,alpha=0.4)
    plt.title(f'{args.dataset}+{args.model} PCA (plain)')
    plt.savefig(f'vis/{args.dataset}_{args.model}_plain.png',dpi=150)
    plt.close()

    # 2) Overlay scatter
    print("Plotting overlay 2D scatter...")
    plt.figure(figsize=(8,6))
    plt.scatter(X2[:,0],X2[:,1],c=mask,cmap='coolwarm',s=10,alpha=0.4)
    ax=plt.gca()
    for idx in sel_corr:
        ax.add_artist(AnnotationBbox(OffsetImage(te_imgs[idx],zoom=0.2),X2[idx],frameon=True,bboxprops=dict(edgecolor='blue',linewidth=0.5)))
    for idx in sel_mis:
        ax.add_artist(AnnotationBbox(OffsetImage(te_imgs[idx],zoom=0.2),X2[idx],frameon=True,bboxprops=dict(edgecolor='red',linewidth=0.5)))
    plt.savefig(f'vis/{args.dataset}_{args.model}_overlay.png',dpi=150)
    plt.close()

    # 3) PCA-3 scatter
    print("Plotting PCA-3 scatter...")
    X3=PCA(3,random_state=0).fit_transform(Xte)
    fig=plt.figure(figsize=(8,6))
    ax3=fig.add_subplot(111,projection='3d')
    ax3.scatter(X3[:,0],X3[:,1],X3[:,2],c=mask,cmap='coolwarm',s=15,alpha=0.6)
    plt.savefig(f'vis/{args.dataset}_{args.model}_3d.png',dpi=150)
    plt.close()
    print("All visualizations saved in ./vis")

if __name__=='__main__': main()