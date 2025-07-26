#!/usr/bin/env bash
set -euo pipefail

# Only DermaMNIST, seeds 0–4, weighted/unweighted NNK voting,
# report mean±std over those five runs.

OUTDOC="all_benchmark_results_dermamnist5.md"
PY="python3 run_benchmark.py"
SEEDS=(0 1 2 3 4)

# DermaMNIST uses full set (n-per-class = 0)
DATASET="dermamnist"
COMMON_FLAGS="--dataset $DATASET --models dino --batch-size 16 --nnk-K 50 --nnk-chunk 64 --n-per-class 0"

# Noise ratios
sym_ratios=(0.0 0.2 0.4 0.6)
asym_ratios=(0.2 0.3 0.4)

# Voting modes
modes=(weighted unweighted)

# Header
cat > "$OUTDOC" <<EOF
# Aggregated Benchmark Results on DermaMNIST (mean±std over seeds 0–4)

_Each block shows the exact command and then the mean±std for all numeric metrics._
EOF

echo -e "\n## Dataset: DermaMNIST\n" >> "$OUTDOC"

# Asymmetric noise
echo -e "### Asymmetric Noise\n" >> "$OUTDOC"
for nr in "${asym_ratios[@]}"; do
  for mode in "${modes[@]}"; do
    OUTCSV="results_${DATASET}_asym_${nr}_${mode}.csv"
    rm -f "$OUTCSV"

    cmd=( $PY $COMMON_FLAGS --noise-pattern asym --noise-ratio "$nr" --nnk-vote-mode "$mode" --outfile "$OUTCSV" )

    # Log command
    {
      echo "**Command:**"
      echo '```bash'
      printf '%q ' "${cmd[@]}"
      echo "--seed ${SEEDS[*]}"
      echo '```'
    } >> "$OUTDOC"

    # Run for each seed (no redirect so you see [STAGE]…)
    for s in "${SEEDS[@]}"; do
      "${cmd[@]}" --seed "$s"
    done

    # Sanity check
    if ! head -1 "$OUTCSV" | grep -q '^Model,'; then
      echo "ERROR: $OUTCSV has no header – aborting." >&2
      exit 1
    fi

    # Compute mean±std
    {
      echo "**Result (mean±std):**"
      python3 - <<PYCODE
import pandas as pd
df = pd.read_csv("$OUTCSV")
num = df.drop(columns=["Model"])
means = num.mean()
stds  = num.std()
row = " | ".join(f"{col}: {means[col]:.3f}±{stds[col]:.3f}" for col in num.columns)
print(f"| {row} |")
PYCODE
      echo
      echo '---'
      echo
    } >> "$OUTDOC"
  done
done

# Symmetric noise
echo -e "### Symmetric Noise\n" >> "$OUTDOC"
for nr in "${sym_ratios[@]}"; do
  for mode in "${modes[@]}"; do
    OUTCSV="results_${DATASET}_sym_${nr}_${mode}.csv"
    rm -f "$OUTCSV"

    cmd=( $PY $COMMON_FLAGS --noise-pattern sym --noise-ratio "$nr" --nnk-vote-mode "$mode" --outfile "$OUTCSV" )

    {
      echo "**Command:**"
      echo '```bash'
      printf '%q ' "${cmd[@]}"
      echo "--seed ${SEEDS[*]}"
      echo '```'
    } >> "$OUTDOC"

    for s in "${SEEDS[@]}"; do
      "${cmd[@]}" --seed "$s"
    done

    if ! head -1 "$OUTCSV" | grep -q '^Model,'; then
      echo "ERROR: $OUTCSV has no header – aborting." >&2
      exit 1
    fi

    {
      echo "**Result (mean±std):**"
      python3 - <<PYCODE
import pandas as pd
df = pd.read_csv("$OUTCSV")
num = df.drop(columns=["Model"])
means = num.mean()
stds  = num.std()
row = " | ".join(f"{col}: {means[col]:.3f}±{stds[col]:.3f}" for col in num.columns)
print(f"| {row} |")
PYCODE
      echo
      echo '---'
      echo
    } >> "$OUTDOC"
  done
done

echo "All runs complete. See $OUTDOC"