#!/bin/bash
# One job-array task per line of a jobs file (see scripts/make_jobs.py).  Run from the repository root:
#   mkdir -p jobs logs results          # logs/ must exist before sbatch (SLURM opens the output file first)
#   python3 scripts/make_jobs.py run_mpo.py --route direct split --n 64 --chi 256 512 --dt 0.05 0.025 > jobs/mpo.txt
#   sbatch --array=1-$(wc -l < jobs/mpo.txt) slurm/array_cpu.sbatch jobs/mpo.txt
# TN runs (run_mpo.py, run_mps.py) are fastest with ONE BLAS thread per task; ED of large rings (L >= 26) profits
# from many threads: submit those with e.g. --cpus-per-task=32 and add "--threads 32" to the ED lines.
# ED memory per block ~ 4 D^2 x 16 B: rings N <= 28 fit in 8G; N = 29, 30: --mem=24G; N = 31-33: --mem=80G.
# Interrupted / timed-out TN tasks resume from their checkpoint when resubmitted with the same line.
#SBATCH --job-name=qlc
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=48:00:00
##SBATCH --partition=your_partition

set -euo pipefail
JOBS=${1:?usage: sbatch --array=1-N slurm/array_cpu.sbatch jobs/file.txt}
source slurm/env.sh
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS")
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} on $(hostname): $CMD"
eval "$CMD"
echo "[$(date)] done"
