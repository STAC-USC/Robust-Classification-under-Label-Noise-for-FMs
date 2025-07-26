# -*- coding: utf-8 -*-
#Monte Carlo simulation of i.i.d. Bernoulli sampling
#and comparison with Hoeffding bound    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Parameters
p_e = 0.2
delta = 0.05
ps_list = [0.05, 0.1, 0.2, 0.5]
T_list = [1, 5, 10, 20, 50]
M = 5000
N = 1000

rng = np.random.default_rng(0)
rows = []
for p_s in ps_list:
    R = max(1, int(round(p_s * N)))
    hoeff_single = np.exp(-2 * R * delta**2)
    for T in T_list:
        count = 0
        for _ in range(M):
            bestf = 1.0
            for _ in range(T):
                xs = rng.random(R) < p_e
                bestf = min(bestf, xs.mean())
            if bestf <= p_e - delta:
                count += 1
        emp = count / M
        bound = 1 - (1 - hoeff_single)**T
        rows.append({
            "p_s": p_s,
            "T": T,
            "empirical_tail": emp,
            "hoeffding_bound": bound
        })

df = pd.DataFrame(rows)

plt.figure()
for T in T_list:
    sub = df[df["T"] == T]
    plt.plot(sub["p_s"], sub["empirical_tail"], marker="o", label=f"emp T={T}")
    plt.plot(sub["p_s"], sub["hoeffding_bound"], linestyle="--", label=f"hoeff T={T}")
plt.xlabel("sampling rate $p_s$")
plt.ylabel(r"$P(M_T \leq p_e - \delta)$")
plt.legend()
plt.tight_layout()
plt.show()
