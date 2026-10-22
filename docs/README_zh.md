# QLC 中文说明

周期 FFD 链 $H=\sum_j Z_jZ_{j+1}X_{j+2}$ 能流 $J=2i\sum_j(h_jh_{j+1}+2h_jh_{j+2})$ 的 Drude 权重数值程序。
由 2026-10-20 程序包（`FFD_periodic_chain_package_20261020`）中与 ED、MPO 演化、纯化 MPS 有关的程序整理重写而成，
结果与原程序一致（测试里直接比较笔记的 $D_N$ 表和原 TN 参考数据）。

## 三个引擎

| 模块 | 内容 | 边界 / 尺寸 |
|---|---|---|
| `qlc.ed` | h-代数约化表示下的全对角化：有限尺寸 Drude 权重 $D_N(\beta)$、关联 $C_N(t)$ | 周期环 `pbc`（块维数 $2^k$：$N=24\to1024$，$N=30\to8192$）与开链 `obc` |
| `qlc.tn.heisenberg` | h-word MPO Heisenberg 演化 + 时间折叠：**直接路线**（演化 $j$）与**拆分路线**（$J=3Q_1+R$，只演化 $R$），$\beta=0$ 与 $\beta>0$ | 无穷链（$n$ 个自旋的开窗口） |
| `qlc.tn.purification` | 自旋基纯化 MPS + TEBD（独立检验），同样两条路线 | $L$ 个自旋的开链 |
| `qlc.tn.mazur` | **静态 Mazur 下界**：准局域荷写成精确 MPS（5 态自动机），$e^{-\beta H}$ 写成 word MPS，算热核 $G_{ab}$ 与 $\langle J,Q_a\rangle$，不做时间演化 | 无穷链（$m$ 个 word 的窗口） |

Drude 权重直接从 $C_J(t)$ 的长时值读出（`qlc.analysis`、`scripts/drude.py`：整周期平台平均、尾巴拟合、
运行平均、dt 的 Richardson 外推、$1/N$ 外推），并与 `qlc.theory` 的闭式（$M^\star=18.6027$、$D_{\rm GHD}(\beta)$、
Mazur 核、Onsager 系数）比较。

## 拆分路线

$Q_1=i\sum_j(h_jh_{j+1}+h_jh_{j+2})$ 守恒且 $\langle Q_1,R\rangle=0$，所以
$C_J(t)=C_R(t)-C_R(0)+C_J(0)$（$\beta=0$ 时 $C_J=18+C_R$）。演化的算符范数 2（而不是 20），相同 $\chi$ 下在长时间
更准：$n=40$、$\chi=64$、$dt=0.05$ 时 $t=3$ 处拆分路线误差 $2\times10^{-4}$，直接路线 $1\times10^{-2}$；
$t\lesssim2$ 两者都由同一 Trotter 误差限制。在有限开链上（纯化、OBC ED）拆分路线还不受 $Q_1$ 部分边界衰减的影响。
两条路线相差 $O(dt^2)$（Trotter 线路只近似守恒 $Q_1$）。

## 常用命令

```bash
pip install -e .                       # numpy, scipy；GPU 另装 torch
pytest -m "not slow"                   # 约 1-2 分钟

python scripts/run_ed.py --L 12 14 16 18 20 --bc pbc --beta 0 0.5 1 2          # D_N(beta)
python scripts/run_ed.py --L 24 --beta 0 1 --tmax 20 --dt 0.05                  # + C_N(t)
python scripts/run_ed.py --L 20 --bc obc --probe local --beta 0 --tmax 3        # 开链，局域探针
python scripts/run_ed.py --L 30 --beta 0 --backend gpu                          # GPU 对角化

python scripts/run_mpo.py --route split  --n 64 --chi 256 --dt 0.05 --tmax 6    # beta = 0，拆分路线
python scripts/run_mpo.py --route direct --n 64 --chi 256 --dt 0.05 --tmax 6    # 直接路线
python scripts/run_mpo.py --route split  --n 64 --beta 1 --chi 256 --chi-rho 128 --tmax 4

python scripts/run_mps.py --route split --L 64 --beta 1 --chi 256 --chi-psi 192 --dt 0.025 --tmax 4 \
    --psi-file results/mps/psi_b1_L64.pkl

python scripts/run_mazur.py --beta 0 0.5 1 2 --chi 256 --backend gpu      # 静态 Mazur 下界（四阶虚时，tau 0.05）
python scripts/drude.py results/mpo/*.json results/ed/pbc_L*_J_total_*.json
python scripts/drude.py --richardson run_dt0.05.json run_dt0.025.json
```

## 上计算节点

* TN 每个进程 **1 个 BLAS 线程**最快（脚本默认 `--threads 1`；本机测试 10 线程反而慢 9 倍），参数扫描用
  作业数组：先 `mkdir -p jobs logs results`，`scripts/make_jobs.py` 生成命令表，
  `sbatch --array=1-N slurm/array_cpu.sbatch jobs/x.txt`。默认输出文件名含参数和参数哈希（不含 tmax），
  扫描不会互相覆盖，重交同一命令即续算。
* ED 每个块内存约 $4D^2\times16$ 字节（$N=30$：4 GiB；$N=31$–33：64 GiB）；扇区缓存与 $\beta$ 无关，加温度不必重算。
* 大环 ED 多线程或 GPU；TN 在 $\chi\gtrsim512$ 时可用 GPU（`--backend gpu`，`slurm/gpu.sbatch`）。
  GPU 后端是 PyTorch 双精度，NVIDIA（CUDA）与海光 DCU / AMD（ROCm、DTK 版 torch，设备名同为 `cuda`）都可用。
  GPU 代码路径在本机用 torch-CPU 测过正确性，GPU 上的速度没有实测。
* 所有 TN 运行定期写检查点，原命令重交即续算（可加大 `--tmax`）；ED 每个扇区单独缓存。
* torch 的 SVD / eigh 结果都会做廉价校验（macOS 上 torch-CPU 的 SVD 对秩亏矩阵会静默出错），失败时自动退回
  `gesvd` 或 CPU 上的 SciPy。

## 注意事项

* $\beta=1$ 时窗口至少 $n\ge32$（$n=24$ 的静态值差 3%）；$\beta=0$、$t\le6$ 用 $n=64$。
* 开链 $D_N^{\rm obc}=0$ 严格成立（$J=i[H,P]$），开链上看 $C(t)$ 的体相平台。
* 环上局域探针会破坏平移对称，程序自动改用全部扇区（不按平移轨道约化）。
* 纯化在 $\beta\sim1$ 用 $dt\le0.025$，并保持 `sym`（三个子格平均）。
* 虚时步长自动调整为 $\beta/(2n)\le$ 给定步长，保证 $\beta$ 精确（不再四舍五入）。
* $\beta>0$ 时 h-word MPO 只接受 K-奇算符（各电流）；K-偶算符（如 H）的关联含不连通部分。
* $\mathfrak L(\beta>0)$ 的公式是模型推广（笔记只推导了 $\beta=0$），旧 TN 数据在 $\beta=1$ 高约 4.5%。

详见 `docs/physics.md`（约定与参考数值）、`docs/methods.md`（算法）、`docs/cluster_gpu.md`（集群与 GPU）。
