# Robust Classification under Noisy Labels: A Geometry-Aware Reliability Framework for Foundation Models

## 📖 Table of Contents

* [About](#about)
* [Features](#features)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Usage](#usage)
* [Outputs](#outputs)
* [Visualizations](#visualizations)
* [Contributing](#contributing)
* [License](#license)
* [References](#references)

## 📄 About

This repository provides a **unified**, **extensible** framework for evaluating and comparing multiple **reliability estimation** methods on foundational vision model embeddings (e.g., CLIP‑ViT, DINOv2) across benchmark datasets.

## ✨ Features

* **Backbone Support**: CLIP‑ViT/32 and DINOv2‑base
* **Datasets**: CIFAR‑10, STL‑10, DermaMNIST (easily add more)
* **Reliability Metrics**:

  * k‑NN confidence
  * NNK graph mass ratio (D/Dc)
  * Per‑class NNK geometric reliability
  * Ensemble NNK reliability (subsample aggregation)
  * Ensemble‑classification NNK (max‑support voting)
  * ANN & WANN baselines
  * NNK‑Means & k‑Means clustering reliability
* **Two‑Stage Pipeline**:

  1. Compute per‑sample reliability score
  2. Classification weighted by reliability
* **Noise Robustness**: inject symmetric label noise up to 40%
* **Visualization**: easy histograms, PCA, and SVM analysis

## 🛠️ Installation

1. **Clone this repo**

   ```bash
   git clone https://github.com/yourusername/foundation-reliability.git
   cd foundation-reliability
   ```
2. **Create & activate a virtualenv**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Run once** to auto-download datasets (CIFAR‑10, STL‑10, DermaMNIST)

## 🚀 Quick Start

```bash
# Benchmark on CIFAR-10 with CLIP and DINO
python run_benchmark.py \
  --dataset cifar10 --models clip dino \
  --n-per-class 100 --batch-size 32 \
  --nnk-K 15 --nnk-chunk 64 \
  --noise-ratio 0.2 --vote-mode weighted \
  --outfile results.csv
```

## 🔧 Usage

```bash
python run_benchmark.py \
  --dataset cifar10 stl10 dermamnist \
  --models clip dino \
  --n-per-class 100 \
  --batch-size 32 \
  --nnk-K 15 \
  --nnk-chunk 64 \
  --no-cache        # force re-extraction
  --noise-ratio 0.2 \
  --vote-mode unweighted \
  --outfile results.csv
```

| Flag            | Description                        |
| --------------- | ---------------------------------- |
| `--dataset`     | `cifar10`, `stl10`, `dermamnist`   |
| `--models`      | `clip`, `dino`                     |
| `--n-per-class` | Samples per class for quick tests  |
| `--batch-size`  | Backbone forward batch size        |
| `--nnk-K`       | NNK neighbor count                 |
| `--nnk-chunk`   | Chunk size for feature extraction  |
| `--no-cache`    | Skip loading cached features       |
| `--noise-ratio` | Label noise fraction (0.0–0.4)     |
| `--vote-mode`   | `weighted` or `unweighted` voting  |
| `--outfile`     | CSV file to append summary results |

## 📂 Outputs

After each run, per-model folders like `nnk_outputs_cifar10_CLIP-ViT/` contain:

```
acc_nnk.npy
y_pred.npy
reliability.npy
reliability_alt.npy
reliability_ens.npy
reliability_nnkmeans.npy
reliability_kmeans.npy
W_te_mean.npy
W_te_std.npy
err_te_mean.npy
err_te_std.npy
ind_te.npy
```

The summary CSV (`results.csv`) is appended with a new row per model.

## 📊 Visualizations

Use the helper script for quick analysis:

```bash
python visualize_nnk_features.py \
  --dataset cifar10 stl10 dermamnist \
  --model CLIP-ViT DINOv2 \
  --out nnk_vis_plots
```



## 🤝 Contributing

1. Fork & clone
2. Create a branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m 'Add my feature'`)
4. Push (`git push origin feature/my-feature`)
5. Open a Pull Request


## 📚 References


