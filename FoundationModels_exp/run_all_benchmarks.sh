#!/usr/bin/env bash
set -euo pipefail

# Script to run benchmarks with both weighted/unweighted NNK voting,
# repeating each configuration for seeds 0,1,2,3,4 and reporting mean±std dev.

OUTDOC="all_benchmark_results8.md"
PY="python3 run_benchmark.py"
# we'll use seeds 0..4
SEEDS=(0 1 2 3 4)

# Datasets and per-class subsampling (0 = full set)
datasets=(cifar10 dermamnist)
declare -A per_class=( [cifar10]=100 [dermamnist]=0 )

# Noise ratios
sym_ratios=(0.0 0.2 0.4 0.6)
declare -A asym_ratios=(
  [cifar10]="0.2 0.3 0.4"
  [dermamnist]="0.2 0.3 0.4"
)

# Voting modes
modes=(weighted unweighted)

# Initialize markdown output
cat > "$OUTDOC" <<EOF
# Aggregated Benchmark Results (mean±std over seeds 0–4)

_Each block shows the exact command and then the mean±std for all numeric metrics._
EOF

for ds in "${datasets[@]}"; do
  echo -e "\n## Dataset: $ds\n" >> "$OUTDOC"

  # build common flags for this dataset
  n_per=${per_class[$ds]}
  if (( n_per > 0 )); then
    COMMON_FLAGS="--dataset $ds --models dino --n-per-class $n_per --batch-size 16 --nnk-K 50 --nnk-chunk 64"
  else
    COMMON_FLAGS="--dataset $ds --models dino --n-per-class 0 --batch-size 16 --nnk-K 50 --nnk-chunk 64"
  fi

  # --- Asymmetric noise ---
  echo -e "### Asymmetric Noise\n" >> "$OUTDOC"
  for nr in ${asym_ratios[$ds]}; do
    for mode in "${modes[@]}"; do
      OUTCSV="results_${ds}_asym_${nr}_${mode}.csv"
      rm -f "$OUTCSV"

      cmd=( $PY $COMMON_FLAGS --noise-pattern asym --noise-ratio "$nr" --nnk-vote-mode "$mode" --outfile "$OUTCSV" )

      # log the command
      {
        echo "**Command:**"
        echo '```bash'
        printf '%q ' "${cmd[@]}"
        echo "--seed ${SEEDS[*]}"
        echo '```'
      } >> "$OUTDOC"

      # run for each seed, _without_ suppressing stdout so you see “[STAGE] …”
      for s in "${SEEDS[@]}"; do
        "${cmd[@]}" --seed "$s"
      done

      # sanity check header
      if ! head -1 "$OUTCSV" | grep -q '^Model,'; then
        echo "ERROR: $OUTCSV has no 'Model' header – aborting."
        exit 1
      fi

      # compute mean±std
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

  # --- Symmetric noise ---
  echo -e "### Symmetric Noise\n" >> "$OUTDOC"
  for nr in "${sym_ratios[@]}"; do
    for mode in "${modes[@]}"; do
      OUTCSV="results_${ds}_sym_${nr}_${mode}.csv"
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
        echo "ERROR: $OUTCSV has no 'Model' header – aborting."
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
done

echo "All runs complete. See $OUTDOC"
