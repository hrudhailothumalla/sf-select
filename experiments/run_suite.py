"""
Master experiment suite for SF-ASS. Each experiment writes JSON to ../results/.
Run individual experiments:  python run_suite.py h1
"""
import sys, os, json, time
import numpy as np
from numpy.random import default_rng
from sklearn.ensemble import RandomForestClassifier
import sfass as S

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)


def save(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print("  saved", name)


def attr_once(X, y, rng, B_trees, cap):
    """One forest fit -> forest attribution phi and single-fit IJ SD."""
    rf = RandomForestClassifier(n_estimators=B_trees, bootstrap=True,
                                max_features="sqrt", n_jobs=-1,
                                random_state=int(rng.integers(1 << 30)))
    rf.fit(X, y)
    pt, Nbi = S.per_tree_oob_importance(rf, X, y, rng, oob_cap=cap)
    phi, V_bc, V_raw, V_mc = S.ij_variance(pt, Nbi)
    return phi, np.sqrt(V_bc), np.sqrt(V_mc), rf


# ---------------------------------------------------------------- H1: validity
def exp_h1(B_trees=150, cap=90, R=25, n=800, p=50, k=8, snr=1.2, seed=1):
    """Does the SINGLE-FIT IJ SD track the TRUE cross-resample SD of phi_j?
    Ground truth: refit on R fresh datasets from the same distribution."""
    print("[H1] variance-estimator validity")
    rng = default_rng(seed)
    # ground-truth sampling distribution of phi across R fresh datasets
    phis = np.zeros((R, p))
    t0 = time.time()
    for r in range(R):
        Xr, yr, mask = S.make_data(n, p, k, default_rng(1000 + r), snr=snr)
        phi_r, _, _, _ = attr_once(Xr, yr, rng, B_trees, cap)
        phis[r] = phi_r
        if r % 5 == 0:
            print(f"   gt refit {r}/{R}  ({time.time()-t0:.0f}s)")
    true_sd = phis.std(axis=0, ddof=1)
    true_mean = phis.mean(axis=0)
    # ONE single fit -> IJ SD
    X0, y0, mask = S.make_data(n, p, k, default_rng(1000), snr=snr)
    phi0, ij_sd, mc_sd, _ = attr_once(X0, y0, rng, B_trees, cap)
    # correlations (across features) between estimated SD and true SD
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    out = dict(
        n=n, p=p, k=k, R=R, B_trees=B_trees,
        corr_ij_true=corr(ij_sd, true_sd),
        corr_mc_true=corr(mc_sd, true_sd),
        corr_ij_true_log=corr(np.log(ij_sd + 1e-9), np.log(true_sd + 1e-9)),
        spearman_ij_true=corr(np.argsort(np.argsort(ij_sd)),
                              np.argsort(np.argsort(true_sd))),
        mean_ratio_ij_true=float(np.median(ij_sd / (true_sd + 1e-12))),
        ij_sd=ij_sd.tolist(), mc_sd=mc_sd.tolist(),
        true_sd=true_sd.tolist(), true_mean=true_mean.tolist(),
        phi0=phi0.tolist(), mask=mask.tolist(),
        seconds=time.time() - t0)
    save("h1_validity.json", out)
    print(f"[H1] corr(IJ_SD, true_SD)={out['corr_ij_true']:.3f}  "
          f"log-corr={out['corr_ij_true_log']:.3f}  "
          f"spearman={out['spearman_ij_true']:.3f}  "
          f"corr(MC_SD,true)={out['corr_mc_true']:.3f}")
    return out


# ------------------------------------------------- H2: cross-cohort reproducibility
def exp_h2(B_trees=130, cap=80, E=8, n=800, p=50, k=8, snr=1.2, ksel=8, seed=2):
    """Select top-ksel features per cohort by raw phi vs normalized W=phi/SD.
    Higher pairwise Jaccard across cohorts = more reproducible selection."""
    print("[H2] cross-cohort selection reproducibility")
    rng = default_rng(seed)
    sel_raw, sel_norm = [], []
    masks = []
    for e in range(E):
        Xe, ye, mask = S.make_data(n, p, k, default_rng(2000 + e), snr=snr)
        phi, ij_sd, mc_sd, _ = attr_once(Xe, ye, rng, B_trees, cap)
        raw_top = np.argsort(phi)[::-1][:ksel]
        norm_top = np.argsort(phi / mc_sd)[::-1][:ksel]  # between-tree normalization (H1 winner)
        sr = np.zeros(p, bool); sr[raw_top] = True
        sn = np.zeros(p, bool); sn[norm_top] = True
        sel_raw.append(sr); sel_norm.append(sn); masks.append(mask)
        print(f"   cohort {e}: raw_true={ (sr & mask).sum() } norm_true={ (sn & mask).sum() }")

    def avg_jaccard(sels):
        js = [S.jaccard(sels[i], sels[j])
              for i in range(len(sels)) for j in range(i + 1, len(sels))]
        return float(np.mean(js))

    def avg_precision(sels, masks):
        return float(np.mean([(s & m).sum() / max(1, s.sum())
                              for s, m in zip(sels, masks)]))
    out = dict(E=E, n=n, p=p, k=k, ksel=ksel,
               jaccard_raw=avg_jaccard(sel_raw),
               jaccard_norm=avg_jaccard(sel_norm),
               precision_raw=avg_precision(sel_raw, masks),
               precision_norm=avg_precision(sel_norm, masks))
    save("h2_reproducibility.json", out)
    print(f"[H2] Jaccard raw={out['jaccard_raw']:.3f}  norm={out['jaccard_norm']:.3f} | "
          f"precision raw={out['precision_raw']:.3f} norm={out['precision_norm']:.3f}")
    return out


# --------------------------------------------------- H3: FDR / power vs CPSS + compute
def exp_h3(reps=10, B_trees=100, cap=55, n=700, p=35, k=7, snr=1.4,
           q=0.2, B_sub=8, seed=3):
    """Single-fit knockoff-FDR (ours, normalized vs raw) vs CPSS multi-fit.
    Measures FDR, power, and wall-clock/# fits."""
    print("[H3] FDR/power and compute vs CPSS")
    agg = {"ours_norm": [], "ours_raw": [], "cpss": []}
    times = {"ours_norm": 0.0, "ours_raw": 0.0, "cpss": 0.0}
    fits = {"ours_norm": 0, "ours_raw": 0, "cpss": 0}
    for rep in range(reps):
        rng = default_rng(3000 + rep)
        X, y, mask = S.make_data(n, p, k, rng, snr=snr)
        for name, norm in [("ours_norm", True), ("ours_raw", False)]:
            t = time.time()
            r = S.sf_ass_knockoff(X, y, rng, B_trees=B_trees, q=q, variance="mc",
                                  normalize=norm, n_shadow=p, oob_cap=cap)
            times[name] += time.time() - t
            fits[name] += 1
            agg[name].append(S.fdr_power(r["sel"], mask))
        t = time.time()
        c = S.cpss(X, y, rng, B_sub=B_sub, B_trees=B_trees, pi_thr=0.6)
        times["cpss"] += time.time() - t
        fits["cpss"] += c["n_fits"]
        agg["cpss"].append(S.fdr_power(c["sel"], mask))
        print(f"   rep {rep} done")
    out = {"q": q, "reps": reps, "n": n, "p": p, "k": k}
    for name in agg:
        a = np.array(agg[name])
        out[name] = dict(FDR=float(a[:, 0].mean()), power=float(a[:, 1].mean()),
                         nsel=float(a[:, 2].mean()),
                         sec_per_run=times[name] / reps,
                         fits_per_run=fits[name] / reps)
    out["compute_ratio_fits"] = out["cpss"]["fits_per_run"] / out["ours_norm"]["fits_per_run"]
    out["compute_ratio_wall"] = out["cpss"]["sec_per_run"] / out["ours_norm"]["sec_per_run"]
    save("h3_fdr_compute.json", out)
    for name in ["ours_norm", "ours_raw", "cpss"]:
        d = out[name]
        print(f"[H3] {name:9s} FDR={d['FDR']:.3f} power={d['power']:.3f} "
              f"nsel={d['nsel']:.1f} fits/run={d['fits_per_run']:.0f} "
              f"sec/run={d['sec_per_run']:.1f}")
    print(f"[H3] compute ratio: {out['compute_ratio_fits']:.0f}x fits, "
          f"{out['compute_ratio_wall']:.0f}x wall")
    return out


# --------------------------------------------------- Scalability (fit-count driven)
def exp_scale(seed=5):
    """Wall-clock of ONE SF-ASS fit vs CPSS (B_sub fits) as n,p grow.
    Uses MDI attribution for the sweep so many sizes are affordable; the
    fit-count ratio (O(1) vs O(B)) is the invariant point."""
    print("[SCALE] runtime scaling")
    rows = []
    for n, p in [(500, 40), (1000, 80), (2000, 160), (4000, 320)]:
        rng = default_rng(seed)
        X, y, mask = S.make_data(n, p, max(5, p // 10), rng, snr=1.2)
        t = time.time()
        rf = RandomForestClassifier(n_estimators=150, n_jobs=-1,
                                    random_state=1).fit(X, y)
        _ = rf.feature_importances_
        t_fit = time.time() - t
        rows.append(dict(n=n, p=p, sfass_fit_s=t_fit,
                         cpss_est_s=t_fit * 24))  # 24 = 2*B_sub(12)
        print(f"   n={n} p={p}: 1 fit={t_fit:.2f}s  -> CPSS(24 fits)~{t_fit*24:.1f}s")
    save("scale.json", {"rows": rows})
    return rows


# --------------------------------------------------- Robustness (label noise)
def exp_robust(B_trees=150, cap=80, n=800, p=40, k=8, seed=6):
    print("[ROBUST] label-noise sensitivity")
    rows = []
    for ln in [0.0, 0.05, 0.1, 0.2]:
        rng = default_rng(seed)
        fdrs, pws = [], []
        for rep in range(3):
            X, y, mask = S.make_data(n, p, k, default_rng(600 + rep), snr=1.3,
                                     label_noise=ln)
            r = S.sf_ass_knockoff(X, y, rng, B_trees=B_trees, q=0.2, variance="mc",
                                  normalize=False, n_shadow=p, oob_cap=cap)
            f, w, _ = S.fdr_power(r["sel"], mask)
            fdrs.append(f); pws.append(w)
        rows.append(dict(label_noise=ln, FDR=float(np.mean(fdrs)),
                         power=float(np.mean(pws))))
        print(f"   label_noise={ln}: FDR={np.mean(fdrs):.3f} power={np.mean(pws):.3f}")
    save("robust.json", {"rows": rows})
    return rows


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which in ("h1", "all"):
        exp_h1()
    if which in ("h2", "all"):
        exp_h2()
    if which in ("h3", "all"):
        exp_h3()
    if which in ("scale", "all"):
        exp_scale()
    if which in ("robust", "all"):
        exp_robust()
    print(f"TOTAL {time.time()-t0:.0f}s")
