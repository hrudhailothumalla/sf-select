"""
E8: Restoring FDR control under correlation via feature clustering.

Per-feature SF-Select fails under correlation (E6/E7). Here we cluster
correlated features and select at the GROUP level: fit one forest on
[X | shadow(X)], cluster the real columns by |correlation|, mirror the same
clustering onto the shadow columns (size-matched null groups), score each group
by jointly-permuting its columns (OOB permutation importance, between-tree
averaged), and apply the same knockoff+ threshold to group scores.

Ground truth at the group level: a real group is RELEVANT if it contains >=1
planted signal feature, else NULL. We report group-level FDR and power
(coverage of relevant groups), which is the honest granularity after clustering.

Writes results/group.json.
"""
import os, json, time
import numpy as np
from numpy.random import default_rng
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestClassifier
import sfass as S
from grid import make_corr_data

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def make_block_data(n, p, rng, n_blocks=8, block_size=5, rho=0.7,
                    sig_blocks=5, snr=1.3):
    """Realistic block correlation: `n_blocks` equicorrelated blocks of
    `block_size`; the first `sig_blocks` blocks each carry ONE signal feature.
    Remaining features are independent nulls. Returns X, y, signal mask."""
    Sigma = np.eye(p)
    for b in range(n_blocks):
        idx = list(range(b * block_size, (b + 1) * block_size))
        for i in idx:
            for j in idx:
                if i != j:
                    Sigma[i, j] = rho
    L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(p))
    X = rng.standard_normal((n, p)) @ L.T
    beta = np.zeros(p)
    signal = [b * block_size for b in range(sig_blocks)]     # one per signal block
    beta[signal] = np.linspace(1.5, 0.8, sig_blocks) * rng.choice([-1, 1], sig_blocks)
    prob = 1 / (1 + np.exp(-snr * (X @ beta)))
    y = (rng.random(n) < prob).astype(int)
    mask = np.zeros(p, bool); mask[signal] = True
    return X, y, mask


def cluster_features(X, corr_thr=0.5):
    """Average-linkage clusters on distance 1-|corr|; merge features with
    |corr| > corr_thr. Returns list of index-lists (clusters of X's columns)."""
    p = X.shape[1]
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C)
    D = 1.0 - np.abs(C)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - corr_thr, criterion="distance")
    return [list(np.where(labels == c)[0]) for c in np.unique(labels)]


def group_importance_per_tree(rf, X, y, groups, rng, oob_cap=80):
    """(B, G) per-tree OOB importance of jointly permuting each group's cols."""
    from joblib import Parallel, delayed
    n = X.shape[0]; B = len(rf.estimators_); G = len(groups)
    Nbi = np.zeros((B, n)); all_idx = np.arange(n)
    seeds = rng.integers(1 << 30, size=B); jobs = []; valid = []

    def one_tree(tree, Xo, yo, seed):
        r = default_rng(seed); no = len(yo)
        if no > oob_cap:
            s = r.choice(no, oob_cap, replace=False); Xo, yo = Xo[s], yo[s]; no = oob_cap
        base = (tree.predict(Xo) == yo).mean()
        out = np.empty(G)
        big = np.repeat(Xo[None, :, :], G, axis=0)          # (G, no, p)
        for gi, g in enumerate(groups):
            for col in g:
                big[gi, :, col] = Xo[r.permutation(no), col]
        preds = tree.predict(big.reshape(G * no, Xo.shape[1])).reshape(G, no)
        out = base - (preds == yo[None, :]).mean(axis=1)
        return out

    for b, tree in enumerate(rf.estimators_):
        counts = np.bincount(rf.estimators_samples_[b], minlength=n); Nbi[b] = counts
        oob = all_idx[counts == 0]
        if len(oob) >= 10:
            jobs.append(delayed(one_tree)(tree, X[oob], y[oob], int(seeds[b]))); valid.append(b)
    outs = Parallel(n_jobs=1)(jobs)
    phi_tree = np.zeros((B, G))
    for b, o in zip(valid, outs):
        phi_tree[b] = o
    return phi_tree


def _shadow_copies(X, rng, n_copies):
    """n_copies independent permuted copies of every column (column-wise perms)."""
    n, p = X.shape
    cols = []
    for _ in range(n_copies):
        for i in range(p):
            cols.append(X[rng.permutation(n), i])
    return np.column_stack(cols)   # shape (n, n_copies*p); block c, feat i at c*p+i


def group_select(X, y, rng, corr_thr=0.5, B_trees=120, q=0.2, oob_cap=70,
                 n_shadow_copies=1):
    """Cluster real cols; mirror onto n_shadow_copies shadow blocks (size-matched
    null groups); group empirical-null BH selection. More copies lower the BH
    p-value floor, which matters when few clusters form."""
    n, p = X.shape
    Xsh = _shadow_copies(X, rng, n_shadow_copies)
    Xs = np.hstack([X, Xsh])
    groups_real = cluster_features(X, corr_thr)
    # mirror clustering onto each shadow block: real i -> col p + c*p + i
    groups_shadow = []
    for c in range(n_shadow_copies):
        base = p + c * p
        groups_shadow += [[base + i for i in g] for g in groups_real]
    all_groups = groups_real + groups_shadow
    rf = RandomForestClassifier(n_estimators=B_trees, bootstrap=True,
                                max_features="sqrt", n_jobs=-1,
                                random_state=int(rng.integers(1 << 30)))
    rf.fit(Xs, y)
    phi_tree = group_importance_per_tree(rf, Xs, y, all_groups, rng, oob_cap)
    phi = phi_tree.mean(axis=0)
    G = len(groups_real)
    Wr, Ws = phi[:G], phi[G:]
    sel = bh_from_shadows(Wr, Ws, q)
    return sel, groups_real


def bh_from_shadows(Wr, Ws, q):
    """Empirical-null Benjamini-Hochberg: p_g = (1+#{shadow >= W_g})/(1+#shadow),
    then BH at level q. Less conservative than knockoff+ for few discoveries."""
    Ws = np.sort(Ws)
    m = len(Wr); mS = len(Ws)
    pvals = np.array([(1 + int((Ws >= w).sum())) / (mS + 1) for w in Wr])
    order = np.argsort(pvals)
    thresh = q * np.arange(1, m + 1) / m
    passed = pvals[order] <= thresh
    if not passed.any():
        return np.zeros(m, bool)
    kmax = np.where(passed)[0].max()
    cutoff = pvals[order][kmax]
    return pvals <= cutoff


def eval_groups(sel, groups, signal_mask):
    """Group FDR/power: a group is relevant if it contains a signal feature."""
    rel = np.array([any(signal_mask[i] for i in g) for g in groups])
    nsel = sel.sum()
    fdr = (sel & ~rel).sum() / max(1, nsel)
    power = (sel & rel).sum() / max(1, rel.sum())
    return fdr, power, int(nsel), int(rel.sum())


if __name__ == "__main__":
    out = {}; t0 = time.time(); REPS = 15
    # (A) block-correlated sweep: GROUP vs PER-FEATURE head-to-head
    print("[E8-A] block-correlated sweep: group vs per-feature (15 seeds)")
    out["synth_block"] = {}
    for rho in [0.0, 0.5, 0.7, 0.9]:
        gf, gp, pf, pp, ng = [], [], [], [], []
        for r in range(REPS):
            rng = default_rng(90000 + r)
            X, y, mask = make_block_data(900, 60, rng, n_blocks=8, block_size=5,
                                         rho=rho, sig_blocks=5, snr=1.4)
            sel, groups = group_select(X, y, rng, corr_thr=0.3, B_trees=100, oob_cap=60)
            f, pw, ns, nr = eval_groups(sel, groups, mask)
            gf.append(f); gp.append(pw); ng.append(len(groups))
            rk = S.sf_ass_knockoff(X, y, rng, B_trees=100, q=0.2, variance="mc",
                                   normalize=False, n_shadow=60, oob_cap=60)
            pfd, ppw, _ = S.fdr_power(rk["sel"], mask); pf.append(pfd); pp.append(ppw)
        out["synth_block"][str(rho)] = dict(
            group_FDR=float(np.mean(gf)), group_FDR_sd=float(np.std(gf)),
            group_power=float(np.mean(gp)),
            perfeat_FDR=float(np.mean(pf)), perfeat_power=float(np.mean(pp)),
            n_groups=float(np.mean(ng)))
        print(f"   rho={rho}: GROUP FDR={np.mean(gf):.3f} pow={np.mean(gp):.3f} | "
              f"PER-FEAT FDR={np.mean(pf):.3f} pow={np.mean(pp):.3f} ({time.time()-t0:.0f}s)")
        json.dump(out, open(os.path.join(RES, "group.json"), "w"), indent=2)

    # (B) breast_cancer semi-synthetic: group vs per-feature
    print("[E8-B] breast_cancer semi-synthetic: group vs per-feature (10 seeds)")
    from real_data import load_real
    Xr, _ = load_real("breast_cancer")
    gf, gp, pf, pp, ng = [], [], [], [], []
    for r in range(10):
        rng = default_rng(95000 + r); rr = default_rng(5000 + r)
        p = Xr.shape[1]; signal = rr.choice(p, 6, replace=False)
        beta = np.zeros(p); beta[signal] = rr.uniform(0.8, 1.6, 6) * rr.choice([-1, 1], 6)
        prob = 1 / (1 + np.exp(-1.2 * (Xr @ beta)))
        y = (rr.random(Xr.shape[0]) < prob).astype(int)
        mask = np.zeros(p, bool); mask[signal] = True
        sel, groups = group_select(Xr, y, rng, corr_thr=0.3, B_trees=120, oob_cap=60)
        f, pw, ns, nr = eval_groups(sel, groups, mask)
        gf.append(f); gp.append(pw); ng.append(len(groups))
        rk = S.sf_ass_knockoff(Xr, y, rng, B_trees=120, q=0.2, variance="mc",
                               normalize=False, n_shadow=p, oob_cap=60)
        pfd, ppw, _ = S.fdr_power(rk["sel"], mask); pf.append(pfd); pp.append(ppw)
    out["breast"] = dict(group_FDR=float(np.mean(gf)), group_FDR_sd=float(np.std(gf)),
                         group_power=float(np.mean(gp)),
                         perfeat_FDR=float(np.mean(pf)), perfeat_power=float(np.mean(pp)),
                         n_groups=float(np.mean(ng)))
    json.dump(out, open(os.path.join(RES, "group.json"), "w"), indent=2)
    print(f"   breast: GROUP FDR={np.mean(gf):.3f} pow={np.mean(gp):.3f} | "
          f"PER-FEAT FDR={np.mean(pf):.3f} pow={np.mean(pp):.3f}")
    print(f"[E8] done {time.time()-t0:.0f}s")
