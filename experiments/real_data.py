"""
Real-data experiments (reviewer request):
  (R1) Dispersion-law validation on REAL data: does single-fit between-tree
       dispersion s_j track the true cross-resample SD of the importance on
       real feature distributions?  (breast_cancer p=30; madelon p=500)
  (R2) Semi-synthetic FDR/power on REAL correlated features: take real X (real
       correlation structure), plant a known sparse label, and measure FDR /
       power for SF-Select vs CPSS. Directly tests the correlation breakdown on
       real data.
Writes results/real.json.
"""
import os, json, time
import numpy as np
from numpy.random import default_rng
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, fetch_openml
import sfass as S

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def load_real(name):
    if name == "breast_cancer":
        d = load_breast_cancer()
        X = StandardScaler().fit_transform(d.data)
        return X, d.target.astype(int)
    if name == "madelon":
        d = fetch_openml("madelon", version=1, as_frame=False, parser="liac-arff")
        X = StandardScaler().fit_transform(d.data.astype(float))
        y = (np.asarray(d.target) == d.target[0]).astype(int)
        return X, y
    raise ValueError(name)


def mean_abs_offdiag_corr(X):
    C = np.corrcoef(X, rowvar=False)
    p = C.shape[0]
    return float((np.abs(C).sum() - p) / (p * (p - 1)))


def attr(X, y, rng, B_trees=150, cap=60):
    rf = RandomForestClassifier(n_estimators=B_trees, bootstrap=True,
                                max_features="sqrt", n_jobs=-1,
                                random_state=int(rng.integers(1 << 30)))
    rf.fit(X, y)
    pt, Nbi = S.per_tree_oob_importance(rf, X, y, rng, oob_cap=cap)
    phi, V_bc, V_raw, V_mc = S.ij_variance(pt, Nbi)
    return phi, np.sqrt(V_mc), np.sqrt(V_bc)     # phi, between-tree SD, IJ SD


def r1_dispersion(name, R=20, B_trees=150, cap=60, sub_frac=0.7, seed=1):
    print(f"[R1] dispersion-law validation on {name}")
    X, y = load_real(name)
    n, p = X.shape
    rng = default_rng(seed)
    # true cross-resample SD via R subsample refits
    phis = np.zeros((R, p))
    t0 = time.time()
    m = int(sub_frac * n)
    for r in range(R):
        idx = default_rng(1000 + r).choice(n, m, replace=False)
        phi_r, _, _ = attr(X[idx], y[idx], rng, B_trees, cap)
        phis[r] = phi_r
        if r % 5 == 0:
            print(f"   refit {r}/{R} ({time.time()-t0:.0f}s)")
    true_sd = phis.std(axis=0, ddof=1)
    # single fit
    idx = default_rng(1000).choice(n, m, replace=False)
    phi0, bt_sd, ij_sd = attr(X[idx], y[idx], rng, B_trees, cap)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    out = dict(dataset=name, n=n, p=p, R=R,
               mean_abs_corr=mean_abs_offdiag_corr(X),
               corr_bt_true=corr(bt_sd, true_sd),
               corr_ij_true=corr(ij_sd, true_sd),
               spearman_bt=corr(np.argsort(np.argsort(bt_sd)),
                                np.argsort(np.argsort(true_sd))),
               seconds=time.time() - t0)
    print(f"   -> corr(BT,true)={out['corr_bt_true']:.3f}  "
          f"corr(IJ,true)={out['corr_ij_true']:.3f}  "
          f"mean|corr|={out['mean_abs_corr']:.3f}")
    return out


def r2_semisynth(name, k=6, reps=10, snr=1.2, B_trees=120, cap=55, q=0.2,
                 B_sub=8, seed=2):
    """Real X, planted sparse logistic label; nulls are the OTHER real (often
    correlated) columns. FDR/power for SF-Select vs CPSS."""
    print(f"[R2] semi-synthetic FDR/power on real {name} (planted label)")
    X, y0 = load_real(name)
    n, p = X.shape
    rng = default_rng(seed)
    sf = {"fdr": [], "pow": []}; cp = {"fdr": [], "pow": []}
    for r in range(reps):
        rr = default_rng(5000 + r)
        signal = rr.choice(p, k, replace=False)
        beta = np.zeros(p); beta[signal] = rr.uniform(0.8, 1.6, k) * rr.choice([-1, 1], k)
        prob = 1 / (1 + np.exp(-snr * (X @ beta)))
        y = (rr.random(n) < prob).astype(int)
        mask = np.zeros(p, bool); mask[signal] = True
        # SF-Select (raw phi + shadows)
        rk = S.sf_ass_knockoff(X, y, rng, B_trees=B_trees, q=q, variance="mc",
                               normalize=False, n_shadow=p, oob_cap=cap)
        f, pw, _ = S.fdr_power(rk["sel"], mask)
        sf["fdr"].append(f); sf["pow"].append(pw)
        # CPSS
        c = S.cpss(X, y, rng, B_sub=B_sub, B_trees=B_trees, pi_thr=0.6)
        cf, cpw, _ = S.fdr_power(c["sel"], mask)
        cp["fdr"].append(cf); cp["pow"].append(cpw)
        print(f"   rep {r}: SF FDR={f:.2f} pow={pw:.2f} | CPSS FDR={cf:.2f} pow={cpw:.2f}")
    return dict(dataset=name, n=n, p=p, k=k, reps=reps,
                mean_abs_corr=mean_abs_offdiag_corr(X),
                sf_FDR=float(np.mean(sf["fdr"])), sf_FDR_sd=float(np.std(sf["fdr"])),
                sf_power=float(np.mean(sf["pow"])),
                cpss_FDR=float(np.mean(cp["fdr"])), cpss_FDR_sd=float(np.std(cp["fdr"])),
                cpss_power=float(np.mean(cp["pow"])))


if __name__ == "__main__":
    out = {}
    out["r1_breast"] = r1_dispersion("breast_cancer", R=20)
    json.dump(out, open(os.path.join(RES, "real.json"), "w"), indent=2)
    out["r2_breast"] = r2_semisynth("breast_cancer", k=6, reps=10)
    json.dump(out, open(os.path.join(RES, "real.json"), "w"), indent=2)
    out["r1_madelon"] = r1_dispersion("madelon", R=12, B_trees=100, cap=40)
    json.dump(out, open(os.path.join(RES, "real.json"), "w"), indent=2)
    print("[REAL] done")
