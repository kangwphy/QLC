# Environment for the batch scripts; edit for your cluster (sourced by array_cpu.sbatch and gpu.sbatch).
# The batch scripts run with `set -euo pipefail`; module / conda scripts often reference unset variables, so the
# user section below runs with `set +u`.
set +u
# --- edit here -----------------------------------------------------------------------------------------------
#   module load anaconda3            # or: module load python/3.10
#   source activate qlc              # conda env with numpy, scipy (and torch for GPU runs)
#   module load cuda/12.1            # NVIDIA;  Hygon DCU / ROCm: module load dtk (torch built for DTK uses 'cuda')
# --------------------------------------------------------------------------------------------------------------
set -u
export PYTHONPATH="${QLC_ROOT:-$(pwd)}/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs results
