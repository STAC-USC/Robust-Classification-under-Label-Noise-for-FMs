# Aggregated Benchmark Results

This document captures **every** run command and its 1×11 summary table.

---

## Dataset: cifar10

### Asymmetric Noise

**Command:**
```bash
python3 run_benchmark.py --dataset cifar10 --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern asym --noise-ratio 0.2         --nnk-vote-mode weighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.958      0.957     0.853 0.091    0.956        0.943        0.956            0.613     0.962    0.963         0.922       0.964

---

**Command:**
```bash
python3 run_benchmark.py --dataset cifar10 --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern asym --noise-ratio 0.2         --nnk-vote-mode unweighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.958      0.957     0.853 0.091    0.954        0.956        0.953            0.478     0.962    0.963         0.952       0.956

---

**Command:**
```bash
python3 run_benchmark.py --dataset cifar10 --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern asym --noise-ratio 0.3         --nnk-vote-mode weighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.947      0.920     0.856 0.090    0.930        0.905        0.932            0.495     0.955    0.945         0.862       0.945

---

**Command:**
```bash
python3 run_benchmark.py --dataset cifar10 --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern asym --noise-ratio 0.3         --nnk-vote-mode unweighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.947      0.920     0.856 0.090    0.938        0.946        0.938            0.481     0.955    0.945         0.915       0.946

---

**Command:**
```bash
python3 run_benchmark.py --dataset cifar10 --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern asym --noise-ratio 0.4         --nnk-vote-mode weighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.904      0.838     0.859 0.082    0.849        0.839        0.852            0.495     0.903    0.862         0.786       0.846

---

**Command:**
```bash
python3 run_benchmark.py --dataset cifar10 --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern asym --noise-ratio 0.4         --nnk-vote-mode unweighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.904      0.838     0.859 0.082    0.906        0.911        0.904            0.471     0.903    0.862         0.758       0.912

---

### Symmetric Noise

**Command:**
```bash
python3 run_benchmark.py --dataset cifar10 --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern sym --noise-ratio 0.0         --nnk-vote-mode weighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.965      0.979     0.851 0.091    0.979        0.980        0.978            0.929     0.964    0.967         0.972       0.976

---

**Command:**
```bash
python3 run_benchmark.py --dataset cifar10 --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern sym --noise-ratio 0.0         --nnk-vote-mode unweighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.965      0.979     0.851 0.091    0.963        0.966        0.963            0.421     0.964    0.967         0.958       0.966

---

**Command:**
```bash
python3 run_benchmark.py --dataset cifar10 --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern sym --noise-ratio 0.2         --nnk-vote-mode weighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.963      0.971     0.855 0.091    0.967        0.962        0.966            0.951     0.961    0.956         0.943       0.964

---

**Command:**
```bash
python3 run_benchmark.py --dataset cifar10 --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern sym --noise-ratio 0.2         --nnk-vote-mode unweighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.963      0.971     0.855 0.091    0.961        0.966        0.961            0.631     0.961    0.956         0.957       0.964

---

**Command:**
```bash
python3 run_benchmark.py --dataset cifar10 --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern sym --noise-ratio 0.4         --nnk-vote-mode weighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.959      0.962     0.869 0.091    0.949        0.896        0.949            0.948     0.956    0.954         0.874       0.949

---

**Command:**
```bash
python3 run_benchmark.py --dataset cifar10 --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern sym --noise-ratio 0.4         --nnk-vote-mode unweighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.959      0.962     0.869 0.091    0.958        0.958        0.958            0.721     0.956    0.954         0.949       0.959

---

**Command:**
```bash
python3 run_benchmark.py --dataset cifar10 --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern sym --noise-ratio 0.6         --nnk-vote-mode weighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.960      0.947     0.889 0.090    0.919        0.761        0.919            0.946     0.964    0.946         0.669       0.908

---

**Command:**
```bash
python3 run_benchmark.py --dataset cifar10 --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern sym --noise-ratio 0.6         --nnk-vote-mode unweighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.960      0.947     0.889 0.090    0.953        0.957        0.953            0.770     0.964    0.946         0.834       0.951

---

## Dataset: dermamnist

### Asymmetric Noise

**Command:**
```bash
python3 run_benchmark.py --dataset dermamnist --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern asym --noise-ratio 0.2         --nnk-vote-mode weighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.375      0.458     0.627 0.026    0.402        0.415        0.411            0.342     0.336    0.388         0.378       0.336

---

**Command:**
```bash
python3 run_benchmark.py --dataset dermamnist --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern asym --noise-ratio 0.2         --nnk-vote-mode unweighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.375      0.458     0.627 0.026    0.349        0.357        0.344            0.216     0.336    0.388         0.369       0.317

---

**Command:**
```bash
python3 run_benchmark.py --dataset dermamnist --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern asym --noise-ratio 0.3         --nnk-vote-mode weighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.351      0.415     0.630 0.015    0.347        0.382        0.359            0.282     0.315    0.357         0.328       0.319

---

**Command:**
```bash
python3 run_benchmark.py --dataset dermamnist --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern asym --noise-ratio 0.3         --nnk-vote-mode unweighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.351      0.415     0.630 0.015    0.338        0.332        0.344            0.253     0.315    0.357         0.301       0.299

---

**Command:**
```bash
python3 run_benchmark.py --dataset dermamnist --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern asym --noise-ratio 0.4         --nnk-vote-mode weighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.299      0.355     0.634 0.002    0.288        0.305        0.292            0.272     0.253    0.288         0.284       0.232

---

**Command:**
```bash
python3 run_benchmark.py --dataset dermamnist --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern asym --noise-ratio 0.4         --nnk-vote-mode unweighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.299      0.355     0.634 0.002    0.293        0.290        0.309            0.207     0.253    0.288         0.255       0.210

---

### Symmetric Noise

**Command:**
```bash
python3 run_benchmark.py --dataset dermamnist --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern sym --noise-ratio 0.0         --nnk-vote-mode weighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.402      0.512     0.624 0.029    0.461        0.483        0.458            0.376     0.344    0.390         0.425       0.384

---

**Command:**
```bash
python3 run_benchmark.py --dataset dermamnist --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern sym --noise-ratio 0.0         --nnk-vote-mode unweighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.402      0.512     0.624 0.029    0.344        0.386        0.338            0.185     0.344    0.390         0.371       0.303

---

**Command:**
```bash
python3 run_benchmark.py --dataset dermamnist --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern sym --noise-ratio 0.2         --nnk-vote-mode weighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.386      0.473     0.625 0.031    0.444        0.440        0.452            0.347     0.346    0.388         0.355       0.398

---

**Command:**
```bash
python3 run_benchmark.py --dataset dermamnist --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern sym --noise-ratio 0.2         --nnk-vote-mode unweighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.386      0.473     0.625 0.031    0.319        0.373        0.324            0.247     0.346    0.388         0.340       0.322

---

**Command:**
```bash
python3 run_benchmark.py --dataset dermamnist --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern sym --noise-ratio 0.4         --nnk-vote-mode weighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.398      0.483     0.630 0.008    0.392        0.365        0.405            0.355     0.359    0.380         0.344       0.328

---

**Command:**
```bash
python3 run_benchmark.py --dataset dermamnist --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern sym --noise-ratio 0.4         --nnk-vote-mode unweighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist   Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.398      0.483     0.630 0.008    0.351        0.373        0.334            0.270     0.359    0.380         0.284       0.313

---

**Command:**
```bash
python3 run_benchmark.py --dataset dermamnist --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern sym --noise-ratio 0.6         --nnk-vote-mode weighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist    Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.313      0.363     0.636 -0.000    0.313        0.282        0.317            0.253     0.270    0.288         0.228       0.253

---

**Command:**
```bash
python3 run_benchmark.py --dataset dermamnist --models dino --n-per-class 100 --batch-size 16 --nnk-K 50 --nnk-chunk 64 --outfile results_n1.csv         --noise-pattern sym --noise-ratio 0.6         --nnk-vote-mode unweighted
```

**Result:**
 Model  Acc_kNN  LogRegAcc  CentDist    Sil  Acc_NNK  Acc_NNK_alt  Acc_NNK_ens  Acc_NNK_ens_ens  Acc_WANN  Acc_ANN  Acc_NNKMeans  Acc_KMeans
DINOv2    0.313      0.363     0.636 -0.000    0.292        0.297        0.297            0.154     0.270    0.288         0.216       0.247

---

