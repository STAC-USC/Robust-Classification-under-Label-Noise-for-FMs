import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
datasets = ["cifar10", "dermamnist"]
noise_types = {
    "asym": {"label": "Asymmetric", "ratios": {"cifar10": [0.0, 0.2, 0.3, 0.4], "dermamnist": [0.0, 0.2, 0.3, 0.4]}},
    "sym":  {"label": "Symmetric",  "ratios": [0.0, 0.2, 0.4, 0.6]},
}
modes = ["weighted", "unweighted"]

# Metrics to plot and their legend labels
#"K_upperbar""Acc_kNN", "Acc_WANN", "Acc_ANN","Acc_NNK_diam", "Acc_NNK_diam_ens","Acc_NNK", "Acc_NNK_ens",
metrics_to_plot = ["Acc_KMeans","Acc_KMeans_v3","Acc_SuperKMeans"]
#"K*","kNN", "WANN", "ANN","NNK_diam","NNK_diam_ens", "NNK_rel",  "NNK_ens",
legend_labels = [ "KMeans", "Kmeans_soft","Kmeans_SupMulti"]

# Assign distinct colors for each metric
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color'][:len(metrics_to_plot)]

# Ensure output directory
outdir = Path("benchmark_plots")
outdir.mkdir(exist_ok=True)

for ds in datasets:
    for noise_key, noise_cfg in noise_types.items():
        ratios = noise_cfg["ratios"][ds] if isinstance(noise_cfg["ratios"], dict) else noise_cfg["ratios"]

        # load symmetric baseline for asym 0
        baseline_sym0 = {}
        if noise_key == 'asym':
            df0 = pd.read_csv(f"results_{ds}_sym_0.0_weighted.csv")
            baseline_sym0 = df0.drop(columns=["Model"]).mean().to_dict()
        
        # Collect data for both modes
        fig, ax = plt.subplots(figsize=(8, 5))
        all_vals = []  # track all mean±std for y-range
        for idx, metric in enumerate(metrics_to_plot):
            color = colors[idx]
            for mode in modes:
                means = []
                stds = []
                for nr in ratios:
                    # use sym0 for asym 0
                    if noise_key == 'asym' and nr == 0.0:
                        mean = baseline_sym0.get(metric, np.nan)
                        std = 0.0
                    else:
                        csv = Path(f"results_{ds}_{noise_key}_{nr}_{mode}.csv")
                        df = pd.read_csv(csv)
                        num = df.drop(columns=["Model"]).apply(pd.to_numeric)
                        mean = num[metric].mean()
                        std = num[metric].std()
                    means.append(mean)
                    stds.append(std)
                all_vals.extend(np.array(means) + np.array(stds))
                all_vals.extend(np.array(means) - np.array(stds))
                linestyle = '-' if mode == 'weighted' else '--'
                ax.plot(ratios, means, linestyle=linestyle, marker='o', color=color,
                        label=f"{legend_labels[idx]} ({mode[0]})")
                ax.fill_between(ratios,
                                np.array(means)-np.array(stds),
                                np.array(means)+np.array(stds),
                                color=color,
                                alpha=0.2)

        # dynamic y-axis limits with margin
        if all_vals:
            y_min, y_max = np.nanmin(all_vals), np.nanmax(all_vals)
            margin = 0.05 * (y_max - y_min)
            ax.set_ylim(max(0, y_min - margin), min(1, y_max + margin))

        ax.set_xlabel("Noise ratio")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{ds.upper()} — {noise_cfg['label']} Noise")
        ax.set_xticks(ratios)
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        out_file = outdir / f"{ds}_{noise_key}_combined_clustering8.png"
        fig.savefig(out_file, dpi=150)
        plt.close(fig)

print(f"Plots saved to: {outdir}")
