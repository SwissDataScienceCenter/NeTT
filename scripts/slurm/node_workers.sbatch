#!/bin/bash
# One node, N workers per GPU, all pulling from the same file-based queue (see nett/experiments/worker.py).
#   sbatch --export=ALL,NETT_EXPERIMENTS="table1 table2",PER_GPU=4 scripts/slurm/node_workers.sbatch
#   (optional: NETT_MATCH='regex on run ids')
# Adapt the #SBATCH lines and the container/venv activation to your site. NETT_DATA / NETT_RESULTS must be set.
#SBATCH --job-name=nett
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=nett_%j.log
set -u
PER_GPU=${PER_GPU:-4}
GPUS=${GPUS:-4}
EXTRA=()
[ -n "${NETT_EXPERIMENTS:-}" ] && EXTRA+=(--experiment $NETT_EXPERIMENTS)
[ -n "${NETT_MATCH:-}" ] && EXTRA+=(--match "$NETT_MATCH")
pids=()
for g in $(seq 0 $((GPUS-1))); do
  for w in $(seq 1 $PER_GPU); do
    CUDA_VISIBLE_DEVICES=$g python3 -m nett worker "${EXTRA[@]}" &
    pids+=($!)
    sleep 5      # stagger start-up so workers do not claim the same run
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
exit $fail
