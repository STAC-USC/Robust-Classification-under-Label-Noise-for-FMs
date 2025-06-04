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

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, accuracy_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
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
# so you can do `from WANN  import run_wann` and `from ANN import run_ann`
sys.path.insert(0, os.path.join(ROOT, "src", "model"))   
# Import noise injection and alternative methods
from src.utils.noise   import inject_noise
from src.model.WANN    import run_wann
from src.model.ANN     import run_ann

# ─────────────────── reproducibility & device & device ────────────────────────────────
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # for feature extraction

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

# ─────────────────── NNK Utilities ────────────────────────────────────────────
# Alternative per-class NNK weighting
# For each class, build NNK on that class subset and normalize weights by the full-set NNK

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

# ─────────────────── NNK Utilities ────────────────────────────────────────────
# Alternative per-class NNK weighting
# For each class, build NNK on that class subset and normalize weights by the full-set NNK


def R_nnk_reliability(Xtr, ytr, top_k, use_gpu=False, R=5):
    """
    Ensemble train-side reliability by averaging NNK reliability over R random 50% subsamples.
    More efficient: compute subset reliability once per run.
    """
    N = Xtr.shape[0]
    sum_rel = np.zeros(N)
    count = np.zeros(N)
    indices = np.arange(N)
    subset_size = N // 2
    for _ in range(R):
        subs = np.random.choice(indices, size=subset_size, replace=False)
        Xs = Xtr[subs]
        ys = ytr[subs]
        rel_s, _, _, _ = compute_nnk_reliability(Xs, ys, top_k, use_gpu)
        sum_rel[subs] += rel_s
        count[subs] += 1
    # handle any indices never included
    never = count == 0
    if never.any():
        # compute full reliability for those
        rel_full, _, _, _ = compute_nnk_reliability(Xtr[never], ytr[never], top_k, use_gpu)
        sum_rel[never] = rel_full
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
            scores = reliability[neigh] * w
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



# ─────────────────── main runner ─────────────────────────────────────────────
def run(dataset: str, models: List[str], outfile: str,
        n_per_class: int, batch_size: int, use_cache: bool,
        K: int, chunk: int,
        noise_pattern: str=None, noise_ratio: float=0.0,vote_mode: str = "weighted"):
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
        lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(Ftr, y_tr_noisy)
        acc_lr  = lr.score(Fte, te_lbls)
        cent = np.mean([np.linalg.norm(Fte[i]-Ftr[y_tr_noisy==te_lbls[i]].mean(0))
                        for i in range(len(te_lbls))])
        sil = silhouette_score(Fte, y_knn)
        # NNK
       # NNK: capture full output dict and save all components
        nnk_res = nnk_classify_and_score(Ftr, y_tr_noisy, Fte, te_lbls, top_k=K,vote_mode=vote_mode)
        (acc_nnk, y_nnk, reliability, W_tr, ind_tr, err_tr,W_te, ind_te, err_te) = (nnk_res['acc_nnk'],
            nnk_res['y_pred'],nnk_res['reliability'],nnk_res['W_tr'],
            nnk_res['ind_tr'],
            nnk_res['err_tr'],
            nnk_res['W_te'],
            nnk_res['ind_te'],
            nnk_res['err_te']
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
        #alternative : ensemble
        rel_ens = R_nnk_reliability(Ftr, y_tr_noisy, top_k=K, use_gpu=False, R=5)
        # classify using ensemble reliability
        nnk_ens_res = nnk_classify_and_score(Ftr, y_tr_noisy, Fte, te_lbls,top_k=K, use_gpu=False,reliability_override=rel_ens,vote_mode=vote_mode)
        acc_nnk_ens = nnk_ens_res['acc_nnk']
        #save ensemble reliability
        np.save(out_dir / "reliability_ens.npy", rel_ens)
        # WANN & ANN
        acc_wann = run_wann(Ftr, y_tr_noisy, Fte, te_lbls)
        acc_ann = run_ann(Ftr, y_tr_noisy, Fte, te_lbls)
        # collect
        rows.append([tag, acc_knn, acc_lr, cent, sil, acc_nnk, acc_nnk_alt, acc_nnk_ens, acc_wann, acc_ann])
    # summary
    df = pd.DataFrame(rows, columns=[
        "Model","Acc_kNN","LogRegAcc","CentDist","Sil",
        "Acc_NNK","Acc_NNK_alt","Acc_NNK_ens","Acc_WANN","Acc_ANN"
    ])
    print(df.to_string(index=False, float_format="%.3f"))
    if outfile:
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

    args = p.parse_args()
    run(args.dataset, args.models, args.outfile,
        args.n_per_class, args.batch_size, not args.no_cache,
        args.nnk_K, args.nnk_chunk,
        args.noise_pattern, args.noise_ratio, args.nnk_vote_mode)
