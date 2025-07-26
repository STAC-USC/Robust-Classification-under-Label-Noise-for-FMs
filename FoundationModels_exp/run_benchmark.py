#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_benchmark.py  – FM adaptability benchmark (CLIP / DINOv2 + DermaMNIST)
==============================================================

• Per‑dataset / per‑model feature caching under ./cache/
• Chunked NNK graph construction with tqdm progress bar (memory‑safe)
• CLI flags:  --dataset, --models, --n-per-class, --batch-size, --no-cache,
  --nnk-K, --nnk-chunk, --noise-pattern, --noise-ratio, --nnk-vote-mode
"""


import os, sys, time, random, argparse
from pathlib import Path
from typing import List

os.environ["TRANSFORMERS_NO_TF"] = "1"
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, accuracy_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.cluster import kmeans_plusplus
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
import clip
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.spatial.distance import cdist
# Add DermaMNIST support
import medmnist
from medmnist import INFO
from PIL import Image
from typing import Tuple
from typing import List, Optional
from numpy.linalg import norm
# Ensure graph_features is available
try:
    from graph_features import graph_features
except ImportError:
    # fallback: append path
    ROOT = Path(__file__).resolve().parent
    sys.path.append(str(ROOT / "../../PyNNK_graph_construction"))
    from graph_features import graph_features
    from faiss_nnk_neighbors_G import nnk_neighbors_G 
sys.path.insert(0, os.path.join(ROOT, "utils"))
# NNK-Means (ec_nnk_means)
sys.path.append(str(ROOT / "../../NNK-Means-OOD-master/src"))
from ec_nnk_means import NNK_Means, kmeans_plusplus as nnk_kmeans_pp
# so you can do `from WANN  import run_wann` and `from ANN import run_ann`
sys.path.insert(0, os.path.join(ROOT, "src", "model"))   
# Import noise injection and alternative methods
from src.utils.noise   import inject_noise
#from src.model.WANN    import run_wann
from src.model.ANN     import run_ann
#from src.model.utils   import WANN 
from src.model.WANN    import WANN
from ec_nnk_means import NNK_Means
from sklearn.cluster import KMeans



# ─────────────────── reproducibility & device & device ────────────────────────────────
#if args.seed is not None:
#    SEED = args.seed
#else:
#    SEED = int(time.time() * 1e6) % (2**32)

##SEED =seed #0
#random.seed(SEED)
#np.random.seed(SEED)
#torch.manual_seed(SEED)
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # for feature extraction

# ─────────────────── dataset helpers ─────────────────────────────────────────
def load_dataset(name: str, n_per_class: int = 100):
    """Return subsampled (train_imgs, train_lbls, test_imgs, test_lbls)."""
    name = name.lower()
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # DermaMNIST branch (with optional subsampling)
    if name in INFO:
        info = INFO[name]
        cls_name = info['python_class']
        DataClass = getattr(medmnist, cls_name)
        tr_ds = DataClass(split='train', download=True)
        te_ds = DataClass(split='test',  download=True)
        # convert numpy arrays to RGB PIL Images
        tr_imgs = [Image.fromarray(img).convert('RGB') for img in tr_ds.imgs]
        te_imgs = [Image.fromarray(img).convert('RGB') for img in te_ds.imgs]
        tr_lbls = tr_ds.labels.flatten()
        te_lbls = te_ds.labels.flatten()
        print(f"[LOAD] MedMNIST ({name}): loaded train={len(tr_imgs)} test={len(te_imgs)}")
        # subsample per class if requested
        if n_per_class > 0:
            def subsample(imgs, lbls):
                buckets, out_imgs, out_lbls = {}, [], []
                for img, lbl in zip(imgs, lbls):
                    buckets.setdefault(lbl, [])
                    if len(buckets[lbl]) < n_per_class:
                        buckets[lbl].append(img)
                for lbl in sorted(buckets):
                    out_imgs.extend(buckets[lbl])
                    out_lbls.extend([lbl]*len(buckets[lbl]))
                return out_imgs, np.array(out_lbls)
            tr_imgs, tr_lbls = subsample(tr_imgs, tr_lbls)
            te_imgs, te_lbls = subsample(te_imgs, te_lbls)
            print(f"[LOAD] MedMNIST ({name}): subsampled train={len(tr_imgs)} test={len(te_imgs)}")
        return tr_imgs, tr_lbls, te_imgs, te_lbls

    # CIFAR-10/STL-10 branch
    if name == "cifar10":
        tr = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
        te = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
    elif name == "stl10":
        tr = datasets.STL10("./data", split="train", download=True, transform=tfm)
        te = datasets.STL10("./data", split="test",  download=True, transform=tfm)
    else:
        raise ValueError(f"Unknown dataset {name}")

    def _sub(ds):
        buckets, imgs, labels = {}, [], []
        for img, lbl in ds:
            buckets.setdefault(lbl, [])
            if len(buckets[lbl]) < n_per_class:
                # ensure PIL Images
                pil = ToPILImage()(img)
                buckets[lbl].append(pil)
        for lbl in sorted(buckets):
            imgs.extend(buckets[lbl]); labels.extend([lbl]*len(buckets[lbl]))
        return imgs, np.array(labels)

    tr_imgs, tr_lbls = _sub(tr)
    te_imgs, te_lbls = _sub(te)
    
    print(f"[LOAD] {name}: train={len(tr_imgs)} test={len(te_imgs)}")
    return tr_imgs, tr_lbls, te_imgs, te_lbls

# ─────────────────── dataloader wrappers ───────────────────────────────────── ─────────────────────────────────────
class ClipDS(torch.utils.data.Dataset):
    def __init__(self, imgs, pre): self.imgs, self.pre = imgs, pre
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i): return self.pre(self.imgs[i])

class DinoDS(torch.utils.data.Dataset):
    def __init__(self, imgs, proc): self.imgs, self.proc = imgs, proc
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i): return self.proc(images=self.imgs[i], return_tensors="pt").pixel_values[0]

# ─────────────────── feature extraction & caching ────────────────────────────
def extract_features(imgs, model_name, split_tag, ds_tag,
                     batch_size=32, use_cache=True, dl_workers=0,
                     force_fp32=False):
    cache_f = Path("cache") / f"cached_{ds_tag}_{model_name}_{split_tag}_{len(imgs)}.npy"
    cache_f.parent.mkdir(exist_ok=True)
    if use_cache and cache_f.exists():
        print(f"[CACHE] Loaded {cache_f}")
        return np.load(cache_f)

    if model_name == "clip":
        model, pre = clip.load("ViT-B/32", device=DEVICE, jit=False)
        model.eval(); half=False
        ds = ClipDS(imgs, pre); bs=batch_size
        encode = lambda xb: model.encode_image(xb.to(DEVICE, non_blocking=True))

    elif model_name == "dino":
        proc  = AutoImageProcessor.from_pretrained("facebook/dinov2-base",use_fast=True)
        model = AutoModel.from_pretrained("facebook/dinov2-base")
        if torch.cuda.is_available() and not force_fp32:
            model = model.to(DEVICE).eval().half(); half=True
        else:
            model = model.cpu().eval(); half=False
        ds = DinoDS(imgs, proc); bs=batch_size*2
        device_batch = DEVICE if torch.cuda.is_available() and not force_fp32 else "cpu"
        encode = lambda xb: model(xb.to(device_batch, non_blocking=True)).last_hidden_state[:,0,:]

    else:
        raise ValueError("model_name must be 'clip' or 'dino'")

    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False,
                                         num_workers=dl_workers, pin_memory=torch.cuda.is_available())
    feats=[]
    with torch.no_grad():
        for xb in tqdm(loader, desc=f"{model_name.upper()} {split_tag}"):
            if half: xb=xb.half()
            feats.append(encode(xb).cpu())
    feats = torch.cat(feats).numpy()
    if use_cache:
        np.save(cache_f, feats)
        print(f"[CACHE] Saved → {cache_f}")
    return feats
#______________________SOFTMAx ------------------------------------------------
def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)
# ─────────────────── NNK Utilities ────────────────────────────────────────────
# Alternative per-class NNK weighting
# For each class, build NNK on that class subset and normalize weights by the full-set NNK
def reliability_power_nnk(w, r):
    """Apply power-based smoothing to NNK weights using reliability r ∈ [0, 1]."""
    w_adj = np.power(w, r)
    w_adj /= np.sum(w_adj) + 1e-10
    return w_adj
    
def per_class_nnk_reliability(Xtr, y_tr_noisy, top_k, use_gpu=False):
    dim = Xtr.shape[1]
    sigma = np.sqrt(dim)
    N_tr = Xtr.shape[0]
    reliability_alt = np.zeros(N_tr)
    # For each training sample, compute ratio
    for i in range(N_tr):
        xi = Xtr[i].reshape(1, -1)
        # full-set NNK from full train to xi
        W_full, ind_full, _ = nnk_neighbors_G(Xtr, xi, top_k=top_k, use_gpu=use_gpu, sigma=sigma)
        feat_full = Xtr[ind_full[0]]
        d_full = cdist(feat_full, xi, 'euclidean').reshape(-1)
        max_full = np.max(d_full) + 1e-12
        # same-class NNK from class subset to xi
        c = y_tr_noisy[i]
        mask = (y_tr_noisy == c)
        Xc = Xtr[mask]
        W_sub, ind_sub, _ = nnk_neighbors_G(Xc, xi, top_k=top_k, use_gpu=use_gpu, sigma=sigma)
        feat_sub = Xc[ind_sub[0]]
        d_sub = cdist(feat_sub, xi, 'euclidean').reshape(-1) if feat_sub.size else np.array([0.])
        max_sub = np.max(d_sub)
        reliability_alt[i] =  max_full/max_sub 
    return reliability_alt

# Ensemble the diameter‐ratio score:
def R_diam_reliability(Xtr, ytr, top_k, use_gpu=False, R=10):
    N = Xtr.shape[0]
    sum_rel = np.zeros(N, dtype=float)
    count   = np.zeros(N, dtype=int)
    indices = np.arange(N)
    subset_size = int(0.5 * N)
    for _ in range(R):
        subs = np.random.choice(indices, size=subset_size, replace=False)
        rel_sub = per_class_nnk_reliability(Xtr[subs], ytr[subs], top_k, use_gpu)
        sum_rel[subs] += rel_sub
        count[subs]   += 1
    # handle any never‑sampled
    never = (count == 0)
    if never.any():
        fallback = per_class_nnk_reliability(Xtr[never], ytr[never], top_k, use_gpu)
        sum_rel[never]  = fallback
        count[never]    = 1
    return sum_rel / count
        

# ─────────────────── NNK Utilities ────────────────────────────────────────────
# Alternative per-class NNK weighting
# For each class, build NNK on that class subset and normalize weights by the full-set NNK

#-------------------- nnkmeans kmeans -------------------------------------------

def nnkmeans_reliability_score(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    atoms: np.ndarray,
    atom_labels: np.ndarray,
    top_k: int = 10,
    use_gpu: bool = False,
    device: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute NNK-Means reliability by extracting reconstruction coefficients
    directly from the trained NNK_Means model using its `get_codes` flag.

    Returns:
      - reliability: ndarray[N] fraction of reconstruction mass on the true class
      - class_weight_dist: ndarray[N, C] normalized distribution of atom weights per class
    """
    # 1) Device setup
    if device is None:
        device = torch.device('cuda') if (use_gpu and torch.cuda.is_available()) else torch.device('cpu')
    else:
        device = torch.device(device)

    # 2) Initialize and load pretrained atoms
    #C = int(atom_labels.max()) + 1
    C = max(int(atom_labels.max()), int(ytr.max())) + 1
    model = NNK_Means(
        n_components=atoms.shape[0],
        n_nonzero_coefs=top_k,
        n_classes=C,
        use_error_based_buffer=False,
        use_residual_update=False
    ).to(device)
    model.initialize_dictionary(
        torch.from_numpy(atoms).to(device),
        torch.from_numpy(atom_labels).to(device)
    )
    model.eval()

    # 3) Forward pass to get dense coefficient matrix
    with torch.no_grad():
        Xt = torch.from_numpy(Xtr).float().to(device)
        y_labels = torch.from_numpy(ytr).long().to(device)
        # request dense codes
        _, _, _, W_full_t = model(
            Xt, y_labels,
            update_cache=True,
            update_dict=False,
            get_codes=True
        )
        W_full = W_full_t.cpu().numpy()  # shape (N, M)

    # 4) Aggregate weights by atom_labels into a per-class distribution
    N = Xtr.shape[0]
    class_weight_dist = np.zeros((N, C), dtype=float)
    for c in range(C):
        mask = (atom_labels == c)
        class_weight_dist[:, c] = W_full[:, mask].sum(axis=1)

    # 5) Normalize each sample's class distribution
    total = class_weight_dist.sum(axis=1, keepdims=True) + 1e-12
    class_weight_dist /= total

    # 6) Extract reliability = mass on the true class
    reliability = class_weight_dist[np.arange(N), ytr]

    return reliability, class_weight_dist




def compute_class_means(Xtr, ytr):
    """
    One centroid per class (the class‐mean).
    Returns:
      centroids: array (C, D)
      labels:    array (C,)
    """
    classes = np.unique(ytr)
    centroids = np.stack([
        Xtr[ytr == c].mean(axis=0)
        for c in classes
    ])
    return centroids, classes

def kmeans_reliability_score(Xtr, ytr, centroids, centroid_labels, eps=1e-12):
    # softmax over −distance to each prototype
    dists = cdist(Xtr, centroids, metric="euclidean")
    probs = softmax(-dists, axis=1)                # shape (n_samples, C)
    inds = np.argmax(probs, axis=1)
    rel = probs[np.arange(len(Xtr)), inds]
    # zero out any that picked the wrong‐label prototype
    rel *= (centroid_labels[inds] == ytr)
    return rel


def supervised_kmeans_reliability_score(
    Xtr, ytr, n_centers_per_class=3
):
    """
    For each class c we fit KMeans(n_centers_per_class) on Xtr[ytr==c].
    Then for each sample i we compute softmax(-distances) to all centroids,
    but only keep the maximum soft‐max weight among centroids of i's true class.
    """

    classes = np.unique(ytr)
    centroids, labels = [], []

    # 1) fit per‐class kmeans
    for c in classes:
        Xc = Xtr[ytr == c]
        km = KMeans(n_clusters=n_centers_per_class, random_state=seed,
                    n_init='auto').fit(Xc)
        centroids.append(km.cluster_centers_)
        labels += [c] * n_centers_per_class

    centroids = np.vstack(centroids)        # shape = (M, D), where M = 3*C
    labels    = np.array(labels)            # shape = (M,)

    # 2) compute all distances and soft‐max weights
    dists   = cdist(Xtr, centroids, 'euclidean')  # (N, M)
    W_all   = softmax(-dists, axis=1)             # (N, M)

    # 3) for each sample i, pick the highest W_all[i,j] among those j with labels[j] == ytr[i]
    rel = np.empty(Xtr.shape[0], dtype=float)
    for i in range(Xtr.shape[0]):
        mask = (labels == ytr[i])        # which centroids belong to sample i's class
        rel[i] = W_all[i, mask].max()    # pick the “closest” (highest soft‐max) same‐class centroid

    return rel
    
def kmeans_v2_reliability_score(
    Xtr, ytr, n_clusters,
    top_k=10, use_gpu=False,
    weight_mode='dist'  # 'nnk' or 'dist'
):
    """
    1) Run k-means on Xtr with n_clusters = #classes.
    2) Compute a soft label distribution per centroid.
    3) Build either:
       - an NNK graph from centroids->Xtr (if weight_mode='nnk'), or
       - a softmax(-distance) matrix (if weight_mode='dist').
    4) For each sample x_k, reliability[k] = sum_j w[k,j] * p_j(ytr[k]).
    """
    # --- a) fit k-means and extract centroids + hard labels ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(Xtr)
    centroids = kmeans.cluster_centers_
    assignments = kmeans.labels_
    C = n_clusters

    # --- b) build probabilistic labels p_j(c) for each centroid j ---
    #    label_probs[j, c] = P(class=c | centroid=j)
    label_probs = np.zeros((C, C), dtype=float)
    for j in range(C):
        members = ytr[assignments == j]
        if len(members) > 0:
            counts = np.bincount(members, minlength=C)
            label_probs[j] = counts / counts.sum()
        # else leave as zeros

    # --- c) build weight matrix W (n_samples x C) ---
    if weight_mode == 'nnk':
        # your original NNK graph:
        sigma = top_k * np.sqrt(Xtr.shape[1])
        W_sparse, inds, _ = nnk_neighbors_G(
            centroids, Xtr,
            top_k=top_k, use_gpu=use_gpu,
            sigma=sigma
        )
        # reconstruct full W
        W = np.zeros((Xtr.shape[0], C), dtype=float)
        for i in range(Xtr.shape[0]):
            for idx, j in enumerate(inds[i]):
                W[i, j] = W_sparse[i, idx]
        # normalize
        W /= (W.sum(axis=1, keepdims=True) + 1e-12)

    elif weight_mode == 'dist':
        # softmax(-euclidean distance) over all centroids
        dists = cdist(Xtr, centroids, metric='euclidean')  # (n_samples, C)
        W = softmax(-dists, axis=1)

    else:
        raise ValueError(f"Unknown weight_mode={weight_mode}")

    # --- d) compute reliability per sample for its true label ---
    #    rel[k] = sum_j W[k,j] * label_probs[j, ytr[k]]
    #    vectorized:
    #    label_probs[:, ytr] has shape (C, n_samples)
    label_prob_for_true = label_probs[:, ytr]            # (C, n_samples)
    rel = np.sum(W * label_prob_for_true.T, axis=1)      # (n_samples,)

    return rel
def kmeans_v3_reliability_score(
    Xtr, ytr, n_clusters,
    top_k=10, use_gpu=False,
    weight_mode='dist'  # 'nnk' or 'dist'
):
    """
    1) Run k-means on Xtr with n_clusters = #classes.
    2) Compute a soft label distribution per centroid.
    3) Build either:
       - an NNK graph from centroids->Xtr (if weight_mode='nnk'), or
       - a softmax(-distance) matrix (if weight_mode='dist').
    4) BUT *only keep the closest* centroid for each sample:
         rel[k] = p_{j*}( ytr[k] )
       where j* = argmin_j dist(x_k, centroid_j).
    """
    # a) Fit k-means
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto').fit(Xtr)
    centroids   = km.cluster_centers_      # (C, D)
    assignments = km.labels_               # (n_samples,)
    C = n_clusters

    # b) Estimate P(class=c | centroid=j)
    label_probs = np.zeros((C, C), dtype=float)
    for j in range(C):
        members = ytr[assignments == j]
        if len(members) > 0:
            counts = np.bincount(members, minlength=C)
            label_probs[j] = counts / counts.sum()

    # c) Compute distances (needed to pick the closest centroid)
    dists = cdist(Xtr, centroids, metric='euclidean')  # (n_samples, C)
    closest_centroid = np.argmin(dists, axis=1)        # (n_samples,)

    # d) Build weight matrix only if needed for 'nnk' (we won't use it for dist)
    if weight_mode == 'nnk':
        # original NNK graph code to get W_sparse and inds...
        sigma = top_k * np.sqrt(Xtr.shape[1])
        W_sparse, inds, _ = nnk_neighbors_G(
            centroids, Xtr,
            top_k=top_k, use_gpu=use_gpu,
            sigma=sigma
        )
        # But again, we'll ignore all but the closest centroid
    elif weight_mode != 'dist':
        raise ValueError(f"Unknown weight_mode={weight_mode}")

    # e) Compute reliability: for each sample k, 
    #    rel[k] = P(class = ytr[k] | centroid = closest_centroid[k])
    rel = label_probs[closest_centroid, ytr]

    return rel

def run_with_nnkmeans_and_kmeans(
    Ftr, y_tr_noisy, Fte, te_lbls,
    atoms, atom_labels,
    top_k=10, use_gpu=False, vote_mode='weighted'):
    C = len(np.unique(y_tr_noisy))

    # 1) NNK‐means reliabilities (unchanged)
    nnk_rel, _ = nnkmeans_reliability_score(
        Ftr, y_tr_noisy, atoms, atom_labels, top_k, use_gpu
    )

    # 2) “Vanilla” k‑means via class‐means
    cm_centroids, cm_labels = compute_class_means(Ftr, y_tr_noisy)
    kmeans_rel = kmeans_reliability_score(
        Ftr, y_tr_noisy, cm_centroids, cm_labels
    )
    #2.5 ) uspervised k-means -3 clusters
    skm_rel = supervised_kmeans_reliability_score(Ftr, y_tr_noisy, n_centers_per_class=3)

    # 3) k‑means v2 (NNK on KMeans centroids)
    kmeans_v2_rel = kmeans_v2_reliability_score(
        Ftr, y_tr_noisy,
        n_clusters=3*C,
        top_k=top_k,
        use_gpu=use_gpu, weight_mode='dist'
    )

    # 4) And finally your three classify‐and‐score calls
    nnk_res = nnk_classify_and_score(
        Ftr, y_tr_noisy, Fte, te_lbls,
        top_k=top_k, use_gpu=use_gpu,
        reliability_override=nnk_rel, vote_mode=vote_mode
    )
    km_res = nnk_classify_and_score(
        Ftr, y_tr_noisy, Fte, te_lbls,
        top_k=top_k, use_gpu=use_gpu,
        reliability_override=kmeans_rel, vote_mode=vote_mode
    )
    km2_res = nnk_classify_and_score(
        Ftr, y_tr_noisy, Fte, te_lbls,
        top_k=top_k, use_gpu=use_gpu,
        reliability_override=kmeans_v2_rel, vote_mode=vote_mode
    )

    skm_res = nnk_classify_and_score(
        Ftr, y_tr_noisy, Fte, te_lbls,
        top_k=top_k, use_gpu=use_gpu,
        reliability_override=skm_rel, vote_mode=vote_mode
    )

    return {
        'nnk':   (nnk_res,   nnk_rel),
        'km':    (km_res,    kmeans_rel),
        'km_v2': (km2_res,   kmeans_v2_rel),
        'sup_km':(skm_res,   skm_rel),
    }
# ─────────────────── NNK Utilities ────────────────────────────────────────────
def R_nnk_reliability(Xtr, ytr, top_k, use_gpu=False, R=10):
    """
    Ensemble train-side reliability by averaging NNK reliability over R random 50% subsamples.
    More efficient: compute subset reliability once per run.
    """
    N = Xtr.shape[0]
    sum_rel = np.zeros(N)
    count = np.zeros(N)
    indices = np.arange(N)
    subset_size = int(0.5* N)
    for _ in range(R):
        subs = np.random.choice(indices, size=subset_size, replace=False)
        Xs = Xtr[subs]
        ys = ytr[subs]
        rel_s =compute_nnk_reliability(Xs, ys, top_k, use_gpu)[0]
        sum_rel[subs] += rel_s
        count[subs] += 1
    # handle any indices never included
      # handle any indices never included
    never = count == 0
    if never.any():
        print(f"[INFO] {never.sum()} samples never sampled — using voting fallback.")
        sum_rel[never] = compute_nnk_reliability(Xtr[never], ytr[never], top_k, use_gpu)[0]
        count[never] = 1
    return sum_rel / count



# ─────────────────── NNK helper via nnk_neighbors_G ─────────────────────────────────────────
# Compute NNK reliability
def compute_nnk_reliability(Xtr, ytr, top_k, use_gpu=False):
     dim = Xtr.shape[1]; sigma = top_k*np.sqrt(dim)
     W_tr, ind_tr, err_tr = nnk_neighbors_G(
         Xtr, Xtr, top_k=top_k, use_gpu=use_gpu, sigma=sigma
     )
     same = (ytr[ind_tr] == ytr[:, None]).astype(float)
     mass_same = (W_tr * same).sum(axis=1)
     mass_all  = W_tr.sum(axis=1) + 1e-12
     reliability = mass_same / mass_all
     return reliability, W_tr, ind_tr, err_tr
     
    
    
# NNK classification: reliability-weighted voting (using Eqns (2) & (1))
def nnk_classify_and_score(Xtr, ytr, Xte, yte, top_k, use_gpu=False, reliability_override: np.ndarray = None, vote_mode: str = "weighted"):
     """
     Perform reliability-weighted classification on test set using NNK.
     If `reliability_override` is provided, use that instead of train-set NNK reliability.
     Returns dict with keys: acc_nnk, y_pred, reliability, W_tr, ind_tr, err_tr, W_te, ind_te, err_te
     """
     # 1) train-side reliability
     if reliability_override is None:
        reliability, W_tr, ind_tr, err_tr = compute_nnk_reliability(
            Xtr, ytr, top_k, use_gpu)
     else:
        reliability = reliability_override
        # skip computing train graph
        W_tr = ind_tr = err_tr = None

     # 2) test-set graph
     dim = Xtr.shape[1]; sigma =  top_k*np.sqrt(dim)
     W_te, ind_te, err_te = nnk_neighbors_G(
         Xtr, Xte, top_k=top_k, use_gpu=use_gpu, sigma=sigma
     )
     # 3) predict via weighted vote
     N_te = Xte.shape[0]
     y_pred = np.empty(N_te, dtype=ytr.dtype)
     for t in range(N_te):
         neigh = ind_te[t]; w = W_te[t]
         if vote_mode == "weighted":
            # weighted by both reliability and edge weight
            r=reliability[neigh]
            scores =(reliability[neigh]*w)
         else:
            # unweighted: rely solely on reliability scores
            scores = reliability[neigh]
         labels = ytr[neigh]
         classes, inv = np.unique(labels, return_inverse=True)
         cs = np.zeros(classes.shape[0])
         np.add.at(cs, inv, scores)
         y_pred[t] = classes[np.argmax(cs)]
     acc_nnk = accuracy_score(yte, y_pred)
     return {
         'acc_nnk': acc_nnk,
         'y_pred': y_pred,
         'reliability': reliability,
         'W_tr': W_tr,
         'ind_tr': ind_tr,
         'err_tr': err_tr,
         'W_te': W_te,
         'ind_te': ind_te,
         'err_te': err_te
     }


def nnk_classify_and_score_mod(Xtr, ytr, Xte, yte, top_k, use_gpu=False, reliability_override: np.ndarray = None, vote_mode: str = "weighted"):
    """
    Perform ensemble reliability-weighted classification on test set using NNK.
    Uses 5 random 50% subsamples to build test-set NNK graphs and aggregate predictions.
    If `reliability_override` is provided, use that instead of train-set NNK reliability.
    Returns dict with keys: acc_nnk, y_pred, reliability, W_tr, ind_tr, err_tr, W_te_mean, W_te_std, ind_te, err_te_mean, err_te_std
    """

    # 1) train-side reliability
    if reliability_override is None:
        reliability, W_tr, ind_tr, err_tr = compute_nnk_reliability(
            Xtr, ytr, top_k, use_gpu)
    else:
        reliability = reliability_override
        W_tr = ind_tr = err_tr = None
    
    #PRUNE top 25% rel
    # how many to keep?
    keep_n = int(0.25 * len(Xtr))
    # sort descending and take indices of the top quarter
    top_idx = np.argsort(reliability)[-keep_n:]
    Xtr = Xtr[top_idx]
    ytr = ytr[top_idx]
    
    N_te = Xte.shape[0]
    vote_counts = np.zeros((N_te, np.max(ytr) + 1))  # Assumes labels in [0, C-1]

    dim = Xtr.shape[1]
    sigma = top_k * np.sqrt(dim)
    R = 5
    subset_size = int(0.8 * Xtr.shape[0])
    
    best_support = np.zeros(N_te, dtype=float)
    y_pred       = np.empty(N_te, dtype=ytr.dtype)
    
    W_te_list = []
    err_te_list = []
    ind_te_set = [set() for _ in range(N_te)]
    
    for _ in range(R):
        # draw a random half of training set
        subs = np.random.choice(len(Xtr), subset_size, replace=False)
        Xs, ys = Xtr[subs], ytr[subs]

        # build NNK graph from Xs to Xte
        W_te, ind_te, err_te = nnk_neighbors_G(
            Xs, Xte, top_k=top_k, use_gpu=use_gpu, sigma=sigma)

        W_te_list.append(W_te)
        err_te_list.append(err_te)

        for t in range(N_te):
            neigh = ind_te[t]
            w     = W_te[t]

            # compute neighbor‐wise scores
            if vote_mode == "weighted":
                scores = reliability[neigh] * w
            else:
                scores = reliability[neigh] 

            labels = ys[neigh]
            classes, inv = np.unique(labels, return_inverse=True)
            cs = np.zeros(len(classes), dtype=float)
            np.add.at(cs, inv, scores)

            # find run‐specific best class and its support
            j_best   = np.argmax(cs)
            support  = cs[j_best]
            chosen_c = classes[j_best]

            # if this run’s support beats previous best, update prediction
            if support > best_support[t]:
                best_support[t] = support
                y_pred[t]       = chosen_c

            # record which neighbors appeared
            ind_te_set[t].update(neigh)

    # final accuracy
    acc_nnk = accuracy_score(yte, y_pred)

    # aggregate stats
    W_te_mean = np.mean(W_te_list, axis=0)
    W_te_std  = np.std(W_te_list,  axis=0)
    err_te_mean = np.mean(err_te_list, axis=0)
    err_te_std  = np.std(err_te_list,  axis=0)

    ind_te_agg = [np.array(sorted(s)) for s in ind_te_set]

    return {
        'acc_nnk': acc_nnk,
        'y_pred': y_pred,
        'reliability': reliability,
        'W_tr': W_tr,
        'ind_tr': ind_tr,
        'err_tr': err_tr,
        'W_te_mean': W_te_mean,
        'W_te_std': W_te_std,
        'ind_te': ind_te_agg,
        'err_te_mean': err_te_mean,
        'err_te_std': err_te_std
    }

# ─────────────────── main runner ─────────────────────────────────────────────
def run(dataset: str, models: List[str], outfile: str,
        n_per_class: int, batch_size: int, use_cache: bool,
        K: int, chunk: int,
        noise_pattern: str=None, noise_ratio: float=0.0,vote_mode: str = "weighted",seed: Optional[int] = None,):
        # Set random seeds if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    tr_imgs, tr_lbls, te_imgs, te_lbls = load_dataset(dataset, n_per_class)
    print(f"[INFO] Dataset={dataset} train={len(tr_imgs)} test={len(te_imgs)}")
    rows = []
    for m in models:
        tag = "CLIP-ViT" if m=="clip" else "DINOv2"
        print("\n" + "="*60 + f"\n[STAGE] {tag}")
        # features
        Ftr = extract_features(tr_imgs, m, "train", dataset, batch_size, use_cache)
        Fte = extract_features(te_imgs, m, "test",  dataset, batch_size, use_cache)
        # normalization
        Ftr = Ftr / np.linalg.norm(Ftr, axis=1, keepdims=True)
        Fte = Fte / np.linalg.norm(Fte, axis=1, keepdims=True)
        y_tr_noisy = tr_lbls
        if noise_pattern and noise_ratio>0:
            y_tr_noisy = inject_noise(dataset, Ftr, tr_lbls, noise_pattern, noise_ratio)
        # basic kNN & LogReg & centdist & sil
        knn = KNeighborsClassifier(K).fit(Ftr, y_tr_noisy)
        y_knn = knn.predict(Fte)
        acc_knn = accuracy_score(te_lbls, y_knn)
        lr = LogisticRegression(max_iter=1000, random_state=seed).fit(Ftr, y_tr_noisy)
        acc_lr  = lr.score(Fte, te_lbls)
        cent = np.mean([np.linalg.norm(Fte[i]-Ftr[y_tr_noisy==te_lbls[i]].mean(0))
                        for i in range(len(te_lbls))])
        sil = silhouette_score(Fte, y_knn)
        # NNK
       # NNK: capture full output dict and save all components
        nnk_res = nnk_classify_and_score(Ftr, y_tr_noisy, Fte, te_lbls, top_k=K,vote_mode=vote_mode)
        (acc_nnk, y_nnk, reliability, W_tr, ind_tr, err_tr,W_te, ind_te, err_te) = (nnk_res['acc_nnk'],
            nnk_res['y_pred'],nnk_res['reliability'],nnk_res['W_tr'],
            nnk_res['ind_tr'],nnk_res['err_tr'],nnk_res['W_te'],nnk_res['ind_te'],nnk_res['err_te']
        )
        # Save NNK outputs for later analysis

        out_dir = Path(f"nnk_outputs_{dataset}_{tag}")
        out_dir.mkdir(exist_ok=True)
        np.save(out_dir / "acc_nnk.npy", np.array([acc_nnk]))
        np.save(out_dir / "y_nnk.npy", y_nnk)
        np.save(out_dir / "reliability.npy", reliability)
        np.save(out_dir / "W_tr.npy", W_tr)
        np.save(out_dir / "ind_tr.npy", ind_tr)
        np.save(out_dir / "err_tr.npy", err_tr)
        np.save(out_dir / "W_te.npy", W_te)
        np.save(out_dir / "ind_te.npy", ind_te)
        np.save(out_dir / "err_te.npy", err_te)
         # Alternative per-class NNK scoring
        rel_alt = per_class_nnk_reliability(Ftr, y_tr_noisy, top_k=K)
        
        # classification using alt reliability
        nnk_alt_res = nnk_classify_and_score(
            Ftr, y_tr_noisy, Fte, te_lbls,
            top_k=K, use_gpu=False,
            reliability_override=rel_alt,vote_mode=vote_mode
        )
        acc_nnk_alt = nnk_alt_res['acc_nnk']
        np.save(out_dir / "reliability_alt.npy", rel_alt)
        rel_diam_ens = R_diam_reliability(Ftr, y_tr_noisy, top_k=K, R=5)
        diam_ens_res = nnk_classify_and_score(
            Ftr, y_tr_noisy, Fte, te_lbls,
            top_k=K, use_gpu=False,
            reliability_override=rel_diam_ens,
            vote_mode=vote_mode
        )
        acc_nnk_diam_ens = diam_ens_res['acc_nnk']
        
        
        #alternative : ensemble
        rel_ens = R_nnk_reliability(Ftr, y_tr_noisy, top_k=K, use_gpu=False, R=5)
        # classify using ensemble reliability
        nnk_ens_res = nnk_classify_and_score(Ftr, y_tr_noisy, Fte, te_lbls,top_k=K, use_gpu=False,reliability_override=rel_ens,vote_mode=vote_mode)
        acc_nnk_ens = nnk_ens_res['acc_nnk']
        #save ensemble reliability
        np.save(out_dir / "reliability_ens.npy", rel_ens)
        
        # 4c) ensemble-classification NNK (new mod)
        nnk_mod_res = nnk_classify_and_score_mod(
             Ftr, y_tr_noisy, Fte, te_lbls,
             top_k=K, use_gpu=False, vote_mode=vote_mode
         )
        acc_nnk_ens_ens = nnk_mod_res['acc_nnk']         # save the new mod outputs
        np.save(out_dir / "W_te_mean_mod.npy", nnk_mod_res['W_te_mean'])
        np.save(out_dir / "W_te_std_mod.npy",  nnk_mod_res['W_te_std'])
        np.save(out_dir / "err_te_mean_mod.npy", nnk_mod_res['err_te_mean'])
        np.save(out_dir / "err_te_std_mod.npy",  nnk_mod_res['err_te_std'])
        # ind_te is a list of arrays; save as object array
        np.save(out_dir / "ind_te_mod.npy",
                np.array(nnk_mod_res['ind_te'], dtype=object))
        # WANN & ANN
        # --- WANN: fit the model so we can extract optimal_k ---
        wann_model = WANN(kmin=11, kmax=51)
        wann_model.fit(Ftr, y_tr_noisy)
        acc_wann = accuracy_score(te_lbls, wann_model.predict(Fte))
        # compute the dataset-level average of the optimal neighborhood sizes:
        K_upperbar = float(np.mean(wann_model.optimal_k))

        acc_ann = run_ann(Ftr, y_tr_noisy, Fte, te_lbls)
        
         # NNK-Means setup

         # NNK-Means initialization
        Xtr_tensor = torch.from_numpy(Ftr).float().cpu()#to(DEVICE)
        ytr_tensor = torch.from_numpy(y_tr_noisy).long().cpu()#to(DEVICE)
        n_components=K*2 
        n_classes=len(np.unique(y_tr_noisy))
        init_idx = nnk_kmeans_pp(Xtr_tensor, n_components)
        nnk_model = NNK_Means(n_components= K,n_nonzero_coefs=2*n_classes, n_classes=len(np.unique(y_tr_noisy)))
        nnk_model.initialize_dictionary(Xtr_tensor[init_idx], ytr_tensor[init_idx])
         # Extract atoms for comparative classification
        atoms = nnk_model.dictionary_atoms.detach().cpu().numpy()
        atom_lbls = nnk_model.atom_labels.detach().cpu().numpy()
        atom_labels = atom_lbls.argmax(axis=1)
        # k-means clustering
        #kmeans = KMeans(n_clusters=n_components, random_state=seed).fit(Ftr)
        #centroids = kmeans.cluster_centers_
        #centroid_labels = np.array([
        #    np.bincount(y_tr_noisy[kmeans.labels_ == i]).argmax()
        #    for i in range(n_components)
        #])
        #n_classes=len(np.unique(y_tr_noisy))
        # Comparative classification
        nnkmeans_res, kmeans_res, kmeans_v2_res, rel_nnkmeans, rel_kmeans, rel_kmeans_v2 = \
            run_with_nnkmeans_and_kmeans(
        Ftr, y_tr_noisy, Fte, te_lbls,
        atoms, atom_labels, top_k=2*n_classes, use_gpu=False,
        vote_mode=vote_mode)
        acc_nnkmeans = nnkmeans_res['acc_nnk']
        acc_kmeans  = kmeans_res['acc_nnk']
        acc_kmeans_v2  = kmeans_v2_res['acc_nnk']
 # ─── Supervised 3‑centroids‑per‑class k‑means ───────────────────────
        skm_rel = supervised_kmeans_reliability_score(
            Ftr, y_tr_noisy, n_centers_per_class=3
        )
        skm_res = nnk_classify_and_score(
            Ftr, y_tr_noisy, Fte, te_lbls,
            top_k=K, use_gpu=False,
            reliability_override=skm_rel, vote_mode=vote_mode
        )
        acc_skm = skm_res['acc_nnk']
        # Save comparative reliability
        out_dir = Path(f"nnk_outputs_{dataset}_{tag}")
        out_dir.mkdir(exist_ok=True)
        np.save(out_dir / "reliability_alt.npy", rel_alt)
        np.save(out_dir / "reliability_ens.npy", rel_ens)
        np.save(out_dir / "reliability_nnkmeans.npy", rel_nnkmeans)
        np.save(out_dir / "reliability_kmeans.npy", rel_kmeans)
        np.save(out_dir / "reliability_kmeans_v2.npy", rel_kmeans_v2)
        np.save(out_dir/"reliability_skm.npy", skm_rel)
        rows.append([
            tag, acc_knn, acc_lr, cent, sil,K_upperbar,
            acc_nnk, acc_nnk_alt,acc_nnk_diam_ens, acc_nnk_ens, acc_nnk_ens_ens,
            acc_wann, acc_ann,
            acc_nnkmeans, acc_kmeans, acc_kmeans_v2,acc_skm
        ])

    df = pd.DataFrame(rows, columns=[
        "Model", "Acc_kNN", "LogRegAcc", "CentDist", "Sil","K_upperbar",
        "Acc_NNK", "Acc_NNK_diam", "Acc_NNK_diam_ens", "Acc_NNK_ens", "Acc_NNK_ens_ens", "Acc_WANN", "Acc_ANN",
        "Acc_NNKMeans", "Acc_KMeans","Acc_KMeans_v2","Acc_SuperKMeans"
    ])
    print(df.to_string(index=False, float_format="%.3f"))

    if outfile:
        if os.path.exists(outfile):
            df.to_csv(outfile, mode='a', index=False, header=False)
        else:
            df.to_csv(outfile, index=False)
def run2(dataset: str, models: List[str], outfile: str,
        n_per_class: int, batch_size: int, use_cache: bool,
        K: int, chunk: int,
        noise_pattern: str=None, noise_ratio: float=0.0,vote_mode: str = "weighted",seed: Optional[int] = None,):
    """
    Same as run(), but only computes and records:
      - Acc_KMeans (softmax over class‑means)
      - Acc_KMeans_v2 (softmax over M centroids)
      - Acc_SuperKMeans (softmax over same‑class centroids)
    """
    # reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # load & feature‐extract
    tr_imgs, tr_lbls, te_imgs, te_lbls = load_dataset(dataset, n_per_class)
    print(f"[INFO] Dataset={dataset} train={len(tr_imgs)} test={len(te_imgs)}")

    rows = []
    for m in models:
        tag = "CLIP-ViT" if m == "clip" else "DINOv2"
        print("\n" + "="*60 + f"\n[STAGE] {tag}")

        # extract & normalize
        Ftr = extract_features(tr_imgs, m, "train", dataset, batch_size, use_cache)
        Fte = extract_features(te_imgs, m, "test",  dataset, batch_size, use_cache)
        Ftr /= np.linalg.norm(Ftr, axis=1, keepdims=True)
        Fte /= np.linalg.norm(Fte, axis=1, keepdims=True)

        # optional noise
        y_tr_noisy = tr_lbls.copy()
        if noise_pattern and noise_ratio > 0:
            y_tr_noisy = inject_noise(dataset, Ftr, tr_lbls, noise_pattern, noise_ratio)

        # 1) vanilla k‑means (class‑means)
        cm_centroids, cm_labels = compute_class_means(Ftr, y_tr_noisy)
        rel_kmeans = kmeans_reliability_score(Ftr, y_tr_noisy, cm_centroids, cm_labels)
        res_km    = nnk_classify_and_score(
            Ftr, y_tr_noisy, Fte, te_lbls,
            top_k=K, use_gpu=False,
            reliability_override=rel_kmeans,
            vote_mode=vote_mode
        )
        acc_kmeans = res_km['acc_nnk']

        # 2) k‑means v3 (softmax over M centroids)
        C = len(np.unique(y_tr_noisy))
        rel_kmeans_v3 = kmeans_v3_reliability_score(
            Ftr, y_tr_noisy,
            n_clusters=3*C,
            weight_mode='dist'
        )
        res_km3   = nnk_classify_and_score(
            Ftr, y_tr_noisy, Fte, te_lbls,
            top_k=K, use_gpu=False,
            reliability_override=rel_kmeans_v3,
            vote_mode=vote_mode
        )
        acc_kmeans_v3 = res_km3['acc_nnk']

        # 3) supervised k‑means (3 centroids per class)
        rel_skm  = supervised_kmeans_reliability_score(Ftr, y_tr_noisy, n_centers_per_class=3)
        res_skm  = nnk_classify_and_score(
            Ftr, y_tr_noisy, Fte, te_lbls,
            top_k=K, use_gpu=False,
            reliability_override=rel_skm,
            vote_mode=vote_mode
        )
        acc_skm = res_skm['acc_nnk']

        # record
        rows.append([tag, acc_kmeans, acc_kmeans_v3, acc_skm])

    # assemble DataFrame
    df = pd.DataFrame(rows, columns=[
        "Model", "Acc_KMeans", "Acc_KMeans_v3", "Acc_SuperKMeans"
    ])
    print(df.to_string(index=False, float_format="%.3f"))

    # save to CSV if requested
    if outfile:
        if os.path.exists(outfile):
            df.to_csv(outfile, mode='a', index=False, header=False)
        else:
            df.to_csv(outfile, index=False)

# ─────────────────── CLI glue ────────────────────────────────────────────────
if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nnk-chunk", type=int, default=256, help="Chunk size for NNK (currently unused)")
    p.add_argument("--dataset", choices=["cifar10","stl10","dermamnist"], default="cifar10")
    p.add_argument("--models", nargs="*", default=["clip"], help="clip dino")
    p.add_argument("--outfile", default="")
    p.add_argument("--n-per-class", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--nnk-K", type=int, default=15)
    p.add_argument("--noise-pattern", choices=["sym","asym","instance"], default=None)
    p.add_argument("--noise-ratio", type=float, default=0.0)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--nnk-vote-mode", choices=["weighted","unweighted"], default="weighted",
                        help="Use reliability-weighted or pure weight voting in NNK.")
    p.add_argument('--seed', type=int, default=None,help='Random seed; if unset, use time-based randomness')

    args = p.parse_args()
    
    if args.seed is None:
        seed = int(time.time() * 1e6) % (2**32)
    else:
        seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run2(args.dataset, args.models, args.outfile,
        args.n_per_class, args.batch_size, not args.no_cache,
        args.nnk_K, args.nnk_chunk,
        args.noise_pattern, args.noise_ratio, args.nnk_vote_mode, args.seed)
