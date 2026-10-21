"""Full diagonalisation of one block and the spectral sums behind the Drude weight and C(t).

For a Hermitian current A and probe B (B = A by default) in the eigenbasis H|n> = E_n|n> of a block:

    W_mn            = A_mn B_nm                                  (= |A_mn|^2 for B = A)
    Drude block sum S_K = sum_{m,n in K} W_mn                    K = degenerate energy block
    static          diag_m = sum_n W_mn = (A B)_mm
    correlation     num(t) = sum_m p_m e^{i E_m t} sum_n W_mn e^{-i E_n t},   p_m = e^{-beta (E_m - E_0)}

so that  Tr(rho A(t) B) = sum over blocks of num(t),  and the infinite-time average keeps only E_m = E_n.
The eigenvector matrix is processed in column chunks, so the memory is one D x D matrix plus O(D x chunk);
W is never stored.  Everything runs on the chosen backend (NumPy or torch/GPU).
"""
import time
import numpy as np

from .algebra import apply_terms


def degenerate_blocks(E, tol=1e-9):
    """block id of every (sorted) eigenvalue; a new block starts where the gap exceeds tol * max(1, max|E|)"""
    E = np.asarray(E)
    cut = np.diff(E) > tol * max(1.0, float(np.abs(E).max()))
    return np.concatenate([[0], np.cumsum(cut)]).astype(np.int64)


def sector_data(rep, A, B=None, ts=None, betas=(0.0,), be=None, tol=1e-9, chunk=1024, log=None):
    """spectral data of one block.

    rep: Representation; A, B: lists of terms (a, b, phase) (B None: B = A); ts: times for C(t) (or None);
    betas: inverse temperatures for the correlation sums.  Returns a dict of NumPy arrays:
      E, block (block id per level), SB (block sums), diag, E0, and if ts is given num[beta] (complex, len(ts)),
      Z[beta] = sum_m p_m, with p_m = exp(-beta (E_m - E0)), E0 = min E.
    """
    from ..backend import get_backend
    be = be or get_backend()
    t0 = time.time()
    D = rep.dim
    Hm = rep.dense(rep.op_terms('H'), be)
    E, V = be.eigh(Hm)
    del Hm
    be.sync()
    t_eigh = time.time() - t0
    En = be.to_numpy(E).real.astype(float)
    E0 = float(En.min())
    bid = degenerate_blocks(En, tol)
    nblk = int(bid[-1]) + 1
    bid_dev = be.asarray(bid, int)
    appA = rep.applier(A, be)
    appB = appA if B is None else rep.applier(B, be)
    diag = be.zeros(D)
    SB = np.zeros(nblk, dtype=complex)
    if ts is not None:
        ts = np.asarray(ts, dtype=float)
        U = be.exp(be.asarray(-1j * np.outer(En, ts)))           # e^{-i E_n t}
        WU = be.zeros((D, len(ts)))
    for s in range(0, D, chunk):
        e = min(D, s + chunk)
        Vc = V[:, s:e]
        AV = apply_terms(appA, Vc, be)
        MA = (AV.conj().T @ V).conj().T                          # (V^dag A V)[:, s:e]
        if B is None:
            Wc = be.abs2(MA) + 0j
        else:
            BV = apply_terms(appB, Vc, be)
            MB = (BV.conj().T @ V).conj().T
            Wc = MA * MB.conj()                                  # A_mn conj(B_mn) = A_mn B_nm
        diag = diag + Wc.sum(1)
        same = bid_dev[:, None] == bid_dev[None, s:e]
        col = be.to_numpy((Wc * same).sum(0))
        np.add.at(SB, bid[s:e], col)
        if ts is not None:
            WU = WU + Wc @ U[s:e]
    out = dict(E=En, block=bid, SB=SB, diag=be.to_numpy(diag), E0=E0, dim=D, weight=rep.weight,
               t_eigh=t_eigh, betas=list(map(float, betas)))
    if ts is not None:
        X = U.conj() * WU                                        # e^{i E_m t} sum_n W_mn e^{-i E_n t}
        out['ts'] = ts
        for b in betas:
            p = be.asarray(np.exp(-b * (En - E0)))
            out[f'num_{b:g}'] = be.to_numpy((p[:, None] * X).sum(0))
            out[f'Z_{b:g}'] = float(np.exp(-b * (En - E0)).sum())
    be.sync()
    out['t_total'] = time.time() - t0
    if log:
        log(f'  {rep.label}: dim {D}, weight {rep.weight}, eigh {t_eigh:.1f}s, total {out["t_total"]:.1f}s, '
            f'{nblk} degenerate blocks (largest {np.bincount(bid).max()})')
    return out


def combine(datas, norm, betas, with_ct=True):
    """thermal averages from a list of sector_data dicts (each counted with its weight).

    D(beta)  = sum_K e^{-beta E_K} S_K / (norm Z)        (infinite-time average of <A(t) B>/norm)
    C0(beta) = sum_m p_m (AB)_mm / (norm Z)
    C(t)     = sum num(t) / (norm Z)                     (if the data contain ts)
    """
    Eg = min(d['E0'] for d in datas)
    res = dict(betas=list(map(float, betas)), D=[], C0=[])
    has_t = with_ct and all('ts' in d for d in datas)
    if has_t:
        res['ts'] = np.asarray(datas[0]['ts'])
        res['C'] = []
    for b in betas:
        Z = Dn = C0 = 0.0
        Ct = 0.0
        for d in datas:
            w = d['weight']
            p = np.exp(-b * (d['E'] - Eg))
            Z += w * p.sum()
            nb = int(d['block'][-1]) + 1
            EK = np.bincount(d['block'], weights=d['E'], minlength=nb) / np.bincount(d['block'], minlength=nb)
            Dn += w * np.sum(np.exp(-b * (EK - Eg)) * d['SB'])
            C0 += w * np.sum(p * d['diag'])
            if has_t:
                key = f'num_{b:g}'
                if key not in d:
                    raise KeyError(f'correlation for beta={b:g} was not computed')
                Ct = Ct + w * np.exp(-b * (d['E0'] - Eg)) * d[key]
        res['D'].append(complex(Dn / (norm * Z)))
        res['C0'].append(complex(C0 / (norm * Z)))
        if has_t:
            res['C'].append(np.asarray(Ct) / (norm * Z))
    return res
