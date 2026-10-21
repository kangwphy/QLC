# Environment for the batch scripts; edit for your cluster (sourced by array_cpu.sbatch and gpu.sbatch).
# Examples:
#   module load anaconda3            # or: module load python/3.10
#   source activate qlc              # conda env with numpy, scipy (and torch for GPU runs)
#   module load cuda/12.1            # NVIDIA;  Hygon DCU / ROCm: module load dtk (torch built for DTK uses 'cuda')
export PYTHONPATH="${QLC_ROOT:-$(pwd)}/src:${PYTHONPATH}"
mkdir -p logs results
