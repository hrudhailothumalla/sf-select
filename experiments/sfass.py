"""
SF-ASS: Single-Fit Attribution Stability Selection
Core library: data generation, attribution, refit-free variance (infinitesimal
jackknife), BH-FDR selection, and the CPSS (bootstrap/subsample) baseline.

All estimators share the SAME base learner (a random forest) and the SAME
attribution (out-of-bag permutation importance) so that the ONLY differences
measured are (a) single-fit analytic variance vs (b) multi-fit resampling
frequency, and the compute ratio.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from numpy.random import default_rng


# --------------------------------------------------------------------------
# 1. Synthetic data: structural, controllable ground truth
# --------------------------------------------------------------------------
def make_data(n, p, k_signal, rng, snr=1.0, corr=0.0, effect_hetero=0.0,
              label_noise=0.0):
    """
    Gaussian design with `p` features. The first `k_signal` features are the
    TRUE signal (nonzero coefficients); the remaining p-k_signal are NULL
    (exactly zero coefficient => any selection of them is a false discovery).

    corr: block correlation among a subset of signal features (to stress
          correlated-feature behaviour).
    effect_hetero: multiplicative random per-draw jitter on coefficients
          (used to make the *sampling* variance of attributions nontrivial).
    Returns X, y (binary), and boolean ground-truth signal mask.
    """
    # Correlated block among first k_signal features (equicorrelation `corr`)
    Sigma = np.eye(p)
    if corr > 0:
        for i in range(k_signal):
            for j in range(k_signal):
                if i != j:
                    Sigma[i, j] = corr
    L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(p))
    X = rng.standard_normal((n, p)) @ L.T

    beta = np.zeros(p)
    # decaying signal strengths so power is a graded quantity, not all-or-none
    strengths = np.linspace(1.5, 0.4, k_signal)
    if effect_hetero > 0:
        strengths = strengths * (1.0 + effect_hetero * rng.standard_normal(k_signal))
    beta[:k_signal] = strengths
    logits = snr * (X @ beta)
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < prob).astype(int)
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y[flip] = 1 - y[flip]
    mask = np.zeros(p, dtype=bool)
    mask[:k_signal] = True
    return X, y, mask


# --------------------------------------------------------------------------
# 2. Attribution: per-tree OOB permutation importance
# --------------------------------------------------------------------------
def _tree_imp(tree, Xo, yo, seed, oob_cap):
    """OOB permutation importance for one tree, all features in ONE predict."""
    rng = default_rng(seed)
    no = len(yo)
    if no > oob_cap:                       # cap OOB set for speed
        sub = rng.choice(no, oob_cap, replace=False)
        Xo, yo = Xo[sub], yo[sub]
        no = oob_cap
    p = Xo.shape[1]
    base = (tree.predict(Xo) == yo).mean()
    # Build one big batch: p permuted copies of Xo, block k has feature k permuted
    big = np.repeat(Xo[None, :, :], p, axis=0)          # (p, no, p)
    for j in range(p):
        big[j, :, j] = Xo[rng.permutation(no), j]
    preds = tree.predict(big.reshape(p * no, p)).reshape(p, no)
    acc = (preds == yo[None, :]).mean(axis=1)
    return base - acc                                    # (p,) importance drops


def per_tree_oob_importance(rf, X, y, rng, oob_cap=200, n_jobs=-1):
    """
    Returns:
      phi_tree : (B_trees, p) per-tree OOB permutation importances
      Nbi      : (B_trees, n) in-bag counts (# times sample i drawn for tree b)
    Byproducts of ONE fitted forest -- no refits. Batched + parallel.
    """
    from joblib import Parallel, delayed
    n, p = X.shape
    B = len(rf.estimators_)
    Nbi = np.zeros((B, n))
    all_idx = np.arange(n)
    jobs, valid = [], []
    seeds = rng.integers(1 << 30, size=B)
    for b, tree in enumerate(rf.estimators_):
        counts = np.bincount(rf.estimators_samples_[b], minlength=n)
        Nbi[b] = counts
        oob = all_idx[counts == 0]
        if len(oob) >= 10:
            jobs.append(delayed(_tree_imp)(tree, X[oob], y[oob], int(seeds[b]), oob_cap))
            valid.append(b)
    out = Parallel(n_jobs=n_jobs, prefer="threads")(jobs)
    phi_tree = np.zeros((B, p))
    for b, imp in zip(valid, out):
        phi_tree[b] = imp
    return phi_tree, Nbi


# --------------------------------------------------------------------------
# 3. Refit-free variance: infinitesimal jackknife (Wager-Hastie-Efron 2014)
# --------------------------------------------------------------------------
def ij_variance(phi_tree, Nbi):
    """
    Forest attribution phi_j = mean_b phi_tree[b,j].
    Returns phi (p,), and THREE variance estimates for each feature j:
       V_ij_bc : bias-corrected infinitesimal jackknife (OUR model variance)
       V_ij_raw: uncorrected IJ
       V_bt    : naive between-tree variance / B (ablation: ignores which
                 samples drove which tree -> the term the IJ correction fixes)
    """
    B, p = phi_tree.shape
    n = Nbi.shape[1]
    phi = phi_tree.mean(axis=0)
    t_centered = phi_tree - phi[None, :]              # (B,p)
    N_centered = Nbi - Nbi.mean(axis=0, keepdims=True)  # (B,n)
    # cov_i,j = (1/B) sum_b N_centered[b,i]*t_centered[b,j]
    cov = (N_centered.T @ t_centered) / B             # (n,p)
    V_ij_raw = (cov ** 2).sum(axis=0)                 # (p,)
    boot_var = (t_centered ** 2).sum(axis=0) / B      # per-feature between-tree var
    V_mc = boot_var / B                               # irreducible Monte-Carlo floor
    V_ij_bc = V_ij_raw - (n / B) * boot_var           # Monte-Carlo bias correction
    # A variance estimate cannot fall below the MC averaging noise of B trees.
    V_ij_bc = np.maximum(V_ij_bc, V_mc)
    return phi, V_ij_bc, np.maximum(V_ij_raw, V_mc), np.maximum(V_mc, 1e-12)


# --------------------------------------------------------------------------
# 4. BH-FDR selection from a one-sided z-statistic
# --------------------------------------------------------------------------
def _norm_sf(z):  # one-sided upper-tail p, no scipy dependency
    from math import erfc, sqrt
    return np.array([0.5 * erfc(zz / sqrt(2.0)) for zz in z])


def bh_select(phi, var, q=0.1):
    z = phi / np.sqrt(var)
    pvals = _norm_sf(z)                 # test H0: phi_j <= 0
    m = len(pvals)
    order = np.argsort(pvals)
    thresh = q * (np.arange(1, m + 1)) / m
    passed = pvals[order] <= thresh
    if not passed.any():
        return np.zeros(m, dtype=bool), pvals, z
    kmax = np.where(passed)[0].max()
    cutoff = pvals[order][kmax]
    return pvals <= cutoff, pvals, z


def add_shadows(X, rng, n_shadow=None):
    """Augment X with 'shadow' features: permuted copies of real columns.
    Shadows are guaranteed nulls and provide a single-fit empirical null."""
    n, p = X.shape
    if n_shadow is None:
        n_shadow = p
    cols = rng.integers(0, p, size=n_shadow)
    Xs = np.column_stack([X[rng.permutation(n), c] for c in cols])
    Xaug = np.hstack([X, Xs])
    is_real = np.array([True] * p + [False] * n_shadow)
    return Xaug, is_real


def knockoff_threshold(W_real, W_shadow, q):
    """Barber-Candes knockoff+ threshold on statistics W (larger = more signal).
    FDP_hat(t) = (1 + #{shadow >= t}) / max(1, #{real >= t}). Pick smallest t
    with FDP_hat <= q; select reals with W_real >= t."""
    cand = np.unique(np.concatenate([W_real, W_shadow]))
    cand = cand[cand > 0]
    best_t = np.inf
    for t in np.sort(cand):
        num = 1 + (W_shadow >= t).sum()
        den = max(1, (W_real >= t).sum())
        if num / den <= q:
            best_t = t
            break
    return best_t


def sf_ass_knockoff(X, y, rng, B_trees=200, q=0.1, variance="mc",
                    normalize=False, n_shadow=None, oob_cap=120, max_features="sqrt"):
    """SINGLE-FIT selection with shadow-calibrated FDR control.
    normalize=True -> statistic W = phi/SD (ours); False -> W = phi (Boruta-like)."""
    Xaug, is_real = add_shadows(X, rng, n_shadow)
    rf = RandomForestClassifier(n_estimators=B_trees, bootstrap=True,
                                max_features=max_features, n_jobs=-1,
                                random_state=int(rng.integers(1 << 30)))
    rf.fit(Xaug, y)
    phi_tree, Nbi = per_tree_oob_importance(rf, Xaug, y, rng, oob_cap=oob_cap)
    phi, V_bc, V_raw, V_mc = ij_variance(phi_tree, Nbi)
    var = {"ij": V_bc, "ij_raw": V_raw, "mc": V_mc}[variance]
    W = phi / np.sqrt(var) if normalize else phi
    Wr, Ws = W[is_real], W[~is_real]
    t = knockoff_threshold(Wr, Ws, q)
    sel = np.zeros(is_real.sum(), dtype=bool)
    if np.isfinite(t):
        sel = Wr >= t
    return dict(sel=sel, W=Wr, W_shadow=Ws, phi=phi[is_real],
                var=var[is_real], t=t, n_fits=1)


def sf_ass(X, y, rng, B_trees=200, q=0.1, variance="ij", max_features="sqrt",
           oob_cap=120):
    """One forest fit -> selection set. Returns dict with selection + diagnostics."""
    rf = RandomForestClassifier(n_estimators=B_trees, bootstrap=True,
                                max_features=max_features, n_jobs=-1,
                                random_state=int(rng.integers(1 << 30)))
    rf.fit(X, y)
    phi_tree, Nbi = per_tree_oob_importance(rf, X, y, rng, oob_cap=oob_cap)
    phi, V_bc, V_raw, V_bt = ij_variance(phi_tree, Nbi)
    var = {"ij": V_bc, "ij_raw": V_raw, "between_tree": V_bt}[variance]
    sel, pvals, z = bh_select(phi, var, q=q)
    return dict(sel=sel, phi=phi, var=var, V_bc=V_bc, V_bt=V_bt,
                pvals=pvals, z=z, n_fits=1)


# --------------------------------------------------------------------------
# 5. CPSS baseline (Shah-Samworth complementary pairs stability selection)
# --------------------------------------------------------------------------
def cpss(X, y, rng, B_sub=50, B_trees=200, pi_thr=0.6, q_select=None,
         max_features="sqrt"):
    """
    Complementary-pairs stability selection. 2*B_sub forest refits.
    Per subsample: fit forest, rank by OOB-style importance (here we use the
    forest's built-in permutation-free importance via mean per-tree OOB perm),
    select top q_select features. Feature j selected overall iff its selection
    frequency across all 2*B_sub fits >= pi_thr.
    Returns selection mask + realized avg #selected/fit (for the S-S bound).
    """
    n, p = X.shape
    if q_select is None:
        q_select = max(5, p // 20)
    freq = np.zeros(p)
    n_fits = 0
    half = n // 2
    for _ in range(B_sub):
        perm = rng.permutation(n)
        for idx in (perm[:half], perm[half:2 * half]):
            rf = RandomForestClassifier(n_estimators=B_trees, bootstrap=True,
                                        max_features=max_features, n_jobs=-1,
                                        random_state=int(rng.integers(1 << 30)))
            rf.fit(X[idx], y[idx])
            pt, Nbi = per_tree_oob_importance(rf, X[idx], y[idx], rng)
            imp = pt.mean(axis=0)
            top = np.argsort(imp)[::-1][:q_select]
            freq[top] += 1
            n_fits += 1
    freq /= n_fits
    sel = freq >= pi_thr
    return dict(sel=sel, freq=freq, n_fits=n_fits, q_select=q_select)


# --------------------------------------------------------------------------
# 6. Metrics
# --------------------------------------------------------------------------
def fdr_power(sel, mask):
    sel_n = sel.sum()
    if sel_n == 0:
        return 0.0, 0.0, 0
    fdr = (sel & ~mask).sum() / sel_n
    power = (sel & mask).sum() / mask.sum()
    return fdr, power, int(sel_n)


def jaccard(a, b):
    a, b = set(np.where(a)[0]), set(np.where(b)[0])
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
