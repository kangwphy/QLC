#!/bin/bash
# Single GPU job (torch backend, double precision).  Run from the repository root, e.g.
#   mkdir -p logs results
#   sbatch slurm/gpu.sbatch python3 scripts/run_ed.py --L 30 --beta 0 0.5 1 2 --backend gpu
#   sbatch slurm/gpu.sbatch python3 scripts/run_mpo.py --route split --n 96 --chi 512 --tmax 6 --backend gpu
# Or a jobs file, one task per GPU:  sbatch --array=1-N slurm/gpu.sbatch --jobs jobs/gpu.txt
#SBATCH --job-name=qlc-gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
##SBATCH --partition=gpu

set -euo pipefail
source slurm/env.sh
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count() and torch.cuda.get_device_name(0))"
if [ "${1:-}" = "--jobs" ]; then
  CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$2")
else
  CMD="$*"
fi
echo "[$(date)] $(hostname): $CMD"
eval "$CMD"
echo "[$(date)] done"
