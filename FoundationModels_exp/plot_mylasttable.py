import os
import matplotlib.pyplot as plt

# 1) Create output directory
os.makedirs('benchmark_plots', exist_ok=True)

# 2) Noise‐level grids
noise_levels_sym  = [0, 20, 40, 60]
noise_levels_asym = [0, 20, 30, 40]

# 3) Color palette
method_colors = {
    'k-NN':        'tab:blue',
    'WANN':        'tab:orange',
    'ANN':         'tab:green',
    'NNK_weights': 'tab:red',
    'NNK_diam':    'tab:brown',
    'k-means_sup':'tab:purple',
    'k-means_usup':'tab:olive',
}

# 4) Data dictionaries (means ± std) without NNK_ens or NNK_diam_ens
# CIFAR-10 Symmetric
# CIFAR‑10 Symmetric
c10_sym_W = {
    'k-NN':         [0.965, 0.966, 0.960, 0.955],
    'WANN':         [0.964, 0.965, 0.959, 0.958],
    'ANN':          [0.967, 0.964, 0.957, 0.944],
    'NNK_weights':  [0.979, 0.973, 0.953, 0.912],
    'NNK_diam':     [0.980, 0.964, 0.910, 0.751],
    'k-means_sup':  [0.975, 0.969, 0.958, 0.950],  # Acc_KMeans
    'k-means_usup': [0.978, 0.971, 0.955, 0.931],  # Acc_KMeans_v3
}
c10_sym_W_std = {
    'k-NN':         [0.000, 0.002, 0.002, 0.005],
    'WANN':         [0.000, 0.004, 0.002, 0.005],
    'ANN':          [0.000, 0.006, 0.003, 0.005],
    'NNK_weights':  [0.000, 0.007, 0.008, 0.005],
    'NNK_diam':     [0.000, 0.003, 0.008, 0.009],
    'k-means_sup':  [0.000, 0.002, 0.004, 0.007],
    'k-means_usup': [0.002, 0.005, 0.007, 0.006],
}

c10_sym_UW = {
    'k-NN':         [0.965, 0.966, 0.960, 0.955],
    'WANN':         [0.964, 0.965, 0.959, 0.958],
    'ANN':          [0.967, 0.964, 0.957, 0.944],
    'NNK_weights':  [0.963, 0.963, 0.958, 0.953],
    'NNK_diam':     [0.966, 0.965, 0.958, 0.956],
    'k-means_sup':  [0.963, 0.965, 0.960, 0.958],  # Acc_KMeans
    'k-means_usup': [0.966, 0.964, 0.960, 0.961],  # Acc_KMeans_v3
}
c10_sym_UW_std = {
    'k-NN':         [0.000, 0.002, 0.002, 0.004],
    'WANN':         [0.000, 0.004, 0.002, 0.005],
    'ANN':          [0.000, 0.006, 0.003, 0.005],
    'NNK_weights':  [0.000, 0.003, 0.003, 0.004],
    'NNK_diam':     [0.000, 0.002, 0.003, 0.004],
    'k-means_sup':  [0.000, 0.002, 0.001, 0.004],
    'k-means_usup': [0.001, 0.001, 0.003, 0.007],
}


# CIFAR‑10 Asymmetric
c10_asym_W = {
    'k-NN':         [0.965, 0.956, 0.942, 0.899],
    'WANN':         [0.964, 0.958, 0.950, 0.907],
    'ANN':          [0.967, 0.955, 0.936, 0.874],
    'NNK_weights':  [0.979, 0.956, 0.923, 0.840],
    'NNK_diam':     [0.980, 0.940, 0.896, 0.821],
    'k-means_sup':  [0.975, 0.967, 0.959, 0.936],  # Acc_KMeans
    'k-means_usup': [0.976, 0.959, 0.933, 0.857],  # Acc_KMeans_v3
}
c10_asym_W_std = {
    'k-NN':         [0.000, 0.007, 0.008, 0.017],
    'WANN':         [0.000, 0.005, 0.009, 0.022],
    'ANN':          [0.000, 0.008, 0.010, 0.030],
    'NNK_weights':  [0.000, 0.008, 0.011, 0.012],
    'NNK_diam':     [0.000, 0.008, 0.013, 0.015],
    'k-means_sup':  [0.000, 0.002, 0.004, 0.008],
    'k-means_usup': [0.000, 0.002, 0.011, 0.012],
}

c10_asym_UW = {
    'k-NN':         [0.965, 0.956, 0.942, 0.899],
    'WANN':         [0.964, 0.958, 0.950, 0.907],
    'ANN':          [0.967, 0.955, 0.936, 0.874],
    'NNK_weights':  [0.963, 0.954, 0.936, 0.896],
    'NNK_diam':     [0.966, 0.955, 0.941, 0.900],
    'k-means_sup':  [0.969, 0.959, 0.954, 0.945],  # Acc_KMeans
    'k-means_usup': [0.969, 0.955, 0.943, 0.909],  # Acc_KMeans_v3
}
c10_asym_UW_std = {
    'k-NN':         [0.000, 0.007, 0.008, 0.017],
    'WANN':         [0.000, 0.005, 0.009, 0.022],
    'ANN':          [0.000, 0.008, 0.010, 0.030],
    'NNK_weights':  [0.000, 0.003, 0.006, 0.012],
    'NNK_diam':     [0.000, 0.006, 0.008, 0.017],
    'k-means_sup':  [0.000, 0.002, 0.005, 0.012],
    'k-means_usup': [0.000, 0.004, 0.004, 0.012],
}


# DermaMNIST Symmetric
# DermaMNIST Symmetric
derm_sym_W = {
    'k-NN':         [0.712, 0.710, 0.710, 0.689],
    'WANN':         [0.710, 0.708, 0.704, 0.688],
    'ANN':          [0.716, 0.712, 0.697, 0.640],
    'NNK_weights':  [0.735, 0.732, 0.715, 0.614],
    'NNK_diam':     [0.759, 0.734, 0.676, 0.498],
    # now replaced:
    'k-means_sup':  [0.766, 0.731, 0.658, 0.474],  # Acc_SuperKMeans (W)
    'k-means_usup':[0.717, 0.718, 0.687, 0.643],  # Acc_KMeans_v3 (W)
}
derm_sym_W_std = {
    'k-NN':         [0.000, 0.003, 0.003, 0.007],
    'WANN':         [0.000, 0.004, 0.003, 0.008],
    'ANN':          [0.000, 0.004, 0.008, 0.008],
    'NNK_weights':  [0.000, 0.005, 0.005, 0.004],
    'NNK_diam':     [0.000, 0.006, 0.007, 0.009],
    'k-means_sup':  [0.001, 0.003, 0.009, 0.011],
    'k-means_usup':[0.002, 0.003, 0.004, 0.010],
}

derm_sym_UW = {
    'k-NN':         [0.712, 0.710, 0.710, 0.689],
    'WANN':         [0.710, 0.708, 0.704, 0.688],
    'ANN':          [0.716, 0.712, 0.697, 0.640],
    'NNK_weights':  [0.704, 0.704, 0.703, 0.698],
    'NNK_diam':     [0.709, 0.708, 0.708, 0.693],
    # now replaced:
    'k-means_sup':  [0.710, 0.710, 0.711, 0.691],  # Acc_SuperKMeans (UW)
    'k-means_usup':[0.689, 0.692, 0.683, 0.693],  # Acc_KMeans_v3 (UW)
}
derm_sym_UW_std = {
    'k-NN':         [0.000, 0.003, 0.003, 0.007],
    'WANN':         [0.000, 0.004, 0.003, 0.008],
    'ANN':          [0.000, 0.004, 0.008, 0.008],
    'NNK_weights':  [0.000, 0.003, 0.001, 0.002],
    'NNK_diam':     [0.000, 0.003, 0.003, 0.006],
    'k-means_sup':  [0.000, 0.003, 0.005, 0.006],
    'k-means_usup':[0.001, 0.004, 0.003, 0.005],
}

# DermaMNIST Asymmetric
derm_asym_W = {
    'k-NN':         [0.712, 0.708, 0.703, 0.661],
    'WANN':         [0.710, 0.704, 0.699, 0.650],
    'ANN':          [0.716, 0.709, 0.683, 0.606],
    'NNK_weights':  [0.735, 0.721, 0.692, 0.600],
    'NNK_diam':     [0.759, 0.717, 0.653, 0.548],
    # now replaced:
    'k-means_sup':  [0.765, 0.715, 0.644, 0.536],  # Acc_SuperKMeans (W)
    'k-means_usup':[0.709, 0.710, 0.690, 0.606],  # Acc_KMeans_v3 (W)
}
derm_asym_W_std = {
    'k-NN':         [0.000, 0.003, 0.005, 0.011],
    'WANN':         [0.000, 0.004, 0.007, 0.013],
    'ANN':          [0.000, 0.005, 0.012, 0.014],
    'NNK_weights':  [0.000, 0.004, 0.004, 0.008],
    'NNK_diam':     [0.000, 0.007, 0.009, 0.007],
    'k-means_sup':  [0.000, 0.006, 0.007, 0.010],
    'k-means_usup':[0.000, 0.003, 0.003, 0.008],
}

derm_asym_UW = {
    'k-NN':         [0.712, 0.708, 0.703, 0.661],
    'WANN':         [0.710, 0.704, 0.699, 0.650],
    'ANN':          [0.716, 0.709, 0.683, 0.606],
    'NNK_weights':  [0.704, 0.699, 0.695, 0.679],
    'NNK_diam':     [0.709, 0.705, 0.700, 0.658],
    # now replaced:
    'k-means_sup':  [0.710, 0.703, 0.703, 0.650],  # Acc_SuperKMeans (UW)
    'k-means_usup':[0.689, 0.688, 0.699, 0.672],  # Acc_KMeans_v3 (UW)
}
derm_asym_UW_std = {
    'k-NN':         [0.000, 0.003, 0.005, 0.011],
    'WANN':         [0.000, 0.004, 0.007, 0.013],
    'ANN':          [0.000, 0.005, 0.012, 0.014],
    'NNK_weights':  [0.000, 0.007, 0.001, 0.006],
    'NNK_diam':     [0.000, 0.003, 0.003, 0.008],
    'k-means_sup':  [0.000, 0.005, 0.007, 0.015],
    'k-means_usup':[0.001, 0.002, 0.003, 0.006],
}



def plot_and_save(key, title, means_W, stds_W, means_UW, stds_UW, noise_levels):
    plt.figure(figsize=(6,4))
    for m, color in method_colors.items():
        if m not in means_W: continue

        # check if this method is identical for W and UW
        same_curve = (means_W[m] == means_UW[m]) and (stds_W[m] == stds_UW[m])

        if same_curve:
            # only plot one line + errorbars
            plt.errorbar(
                noise_levels, means_W[m], yerr=stds_W[m],
                label=m,
                color=color, linestyle='-', marker='o', capsize=3
            )
        else:
        # weighted
            plt.errorbar(
                noise_levels, means_W[m], yerr=stds_W[m],
                label=f"{m} (W)", color=color,
                linestyle='-', marker='o', capsize=3
            )
         # unweighted
            plt.errorbar(
                noise_levels, means_UW[m], yerr=stds_UW[m],
                label=f"{m} (UW)", color=color,
                linestyle='--', marker='s', capsize=3
            )
    plt.title(title)
    plt.xlabel("Noise ratio (%)")
    plt.ylabel("Accuracy")
    plt.xticks(noise_levels)
    # auto‐scale y
    all_vals = []
    for m in means_W:
        all_vals.extend(means_W[m])
        all_vals.extend(means_UW[m])
    #all_vals = sum([means_W[m] + means_UW[m] for m in means_W], [])
    ymin, ymax = min(all_vals), max(all_vals)
    plt.ylim(ymin-0.01, ymax+0.01)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    #plt.savefig(f"./benchmark_plots/{key}.png")
       # <- save as EPS here
    outpath = f"./benchmark_plots/{key}8.eps"
    plt.savefig(outpath, format='eps')
    plt.close()
    print("Saved:", outpath)

# 5) Generate all four
plot_and_save('cifar10_sym',  "CIFAR‑10: symmetric",  c10_sym_W,  c10_sym_W_std,  c10_sym_UW,  c10_sym_UW_std,  noise_levels_sym)
plot_and_save('cifar10_asym', "CIFAR‑10: asymmetric", c10_asym_W, c10_asym_W_std, c10_asym_UW, c10_asym_UW_std, noise_levels_asym)
plot_and_save('derma_sym',    "DermaMNIST: symmetric", derm_sym_W, derm_sym_W_std, derm_sym_UW, derm_sym_UW_std, noise_levels_sym)
plot_and_save('derma_asym',   "DermaMNIST: asymmetric",derm_asym_W,derm_asym_W_std,derm_asym_UW,derm_asym_UW_std,noise_levels_asym)
