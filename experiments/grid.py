"""
Expanded evidence for the reviewer critique:
  (1) q-calibration: realized FDR vs target q (single fit -> all q, cheap).
  (2) correlated-feature stress test: equicorrelation block spanning signals
      AND nulls (the regime the paper admits is hard).
  (3) size breadth: several (n,p).
All use SF-Select with the RAW-phi statistic (the headline selector) and
>=30 repetitions. Writes results/grid.json.
"""
import os, json, time
import numpy as np
from numpy.random import default_rng
from sklearn.ensemble import RandomForestClassifier
import sfass as S

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def make_corr_data(n, p, k, rng, snr=1.3, rho=0.0, corr_block=None):
    """Equicorrelation block over the first `corr_block` features (>=k), so some
    NULL features are correlated with signals -- an FDR stressor."""
    if corr_block is None:
        corr_block = min(p, 2 * k)
    Sigma = np.eye(p)
    if rho > 0:
        Sigma[:corr_block, :corr_block] = rho
        np.fill_diagonal(Sigma, 1.0)
    L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(p))
    X = rng.standard_normal((n, p)) @ L.T
    beta = np.zeros(p)
    beta[:k] = np.linspace(1.5, 0.5, k)
    prob = 1.0 / (1.0 + np.exp(-snr * (X @ beta)))
    y = (rng.random(n) < prob).astype(int)
    mask = np.zeros(p, bool); mask[:k] = True
    return X, y, mask


def fit_stats(X, y, rng, B_trees=110, cap=60):
    """One fit on [X | shadows]; return real & shadow raw-phi importances."""
    Xaug, is_real = S.add_shadows(X, rng, n_shadow=X.shape[1])
    rf = RandomForestClassifier(n_estimators=B_trees, bootstrap=True,
                                max_features="sqrt", n_jobs=-1,
                                random_state=int(rng.integers(1 << 30)))
    rf.fit(Xaug, y)
    pt, Nbi = S.per_tree_oob_importance(rf, Xaug, y, rng, oob_cap=cap)
    phi, _, _, _ = S.ij_variance(pt, Nbi)
    return phi[is_real], phi[~is_real]


def eval_at_q(Wr, Ws, mask, q):
    t = S.knockoff_threshold(Wr, Ws, q)
    sel = Wr >= t if np.isfinite(t) else np.zeros_like(Wr, bool)
    return S.fdr_power(sel, mask)


def cell(n, p, k, rho, reps, qs, seed0, snr=1.4):
    out = {q: {"fdr": [], "pow": [], "nsel": []} for q in qs}
    for r in range(reps):
        rng = default_rng(seed0 + r)
        X, y, mask = make_corr_data(n, p, k, rng, snr=snr, rho=rho)
        Wr, Ws = fit_stats(X, y, rng)
        for q in qs:
            f, pw, ns = eval_at_q(Wr, Ws, mask, q)
            out[q]["fdr"].append(f); out[q]["pow"].append(pw); out[q]["nsel"].append(ns)
    summ = {}
    for q in qs:
        summ[str(q)] = dict(
            FDR=float(np.mean(out[q]["fdr"])), FDR_sd=float(np.std(out[q]["fdr"])),
            power=float(np.mean(out[q]["pow"])), power_sd=float(np.std(out[q]["pow"])),
            nsel=float(np.mean(out[q]["nsel"])))
    return summ


if __name__ == "__main__":
    t0 = time.time()
    reps = 30
    qs = [0.05, 0.1, 0.2]
    results = {"reps": reps, "qs": qs}

    print("[GRID] q-calibration + correlation sweep (n=800,p=60,k=8)")
    results["corr_sweep"] = {}
    for rho in [0.0, 0.3, 0.6, 0.9]:
        results["corr_sweep"][str(rho)] = cell(800, 60, 8, rho, reps, qs, 10000)
        r = results["corr_sweep"][str(rho)]["0.2"]
        print(f"   rho={rho}: q=0.2 -> FDR={r['FDR']:.3f}+-{r['FDR_sd']:.3f} "
              f"power={r['power']:.3f}  ({time.time()-t0:.0f}s)")
        json.dump(results, open(os.path.join(RES, "grid.json"), "w"), indent=2)

    print("[GRID] size breadth (rho=0.3, q sweep)")
    results["size_sweep"] = {}
    for (n, p, k) in [(500, 40, 6), (1000, 100, 10), (2000, 200, 15)]:
        key = f"n{n}_p{p}"
        results["size_sweep"][key] = cell(n, p, k, 0.3, reps, qs, 20000)
        r = results["size_sweep"][key]["0.2"]
        print(f"   {key}: q=0.2 -> FDR={r['FDR']:.3f}+-{r['FDR_sd']:.3f} "
              f"power={r['power']:.3f}  ({time.time()-t0:.0f}s)")
        json.dump(results, open(os.path.join(RES, "grid.json"), "w"), indent=2)

    print(f"[GRID] done {time.time()-t0:.0f}s")
