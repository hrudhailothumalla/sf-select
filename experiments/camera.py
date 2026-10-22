"""
Camera-ready re-run addressing reviewer feedback. One UNIFIED config across all
synthetic experiments (so rho=0 numbers are consistent), PAIRED shadows
(one per feature), MORE seeds, ERROR BARS, CPSS with a standard refit count,
and H1 with within-group + rank correlations + a bootstrap CI.
Writes results/camera.json incrementally.
"""
import os, json, time
import numpy as np
from numpy.random import default_rng
from sklearn.ensemble import RandomForestClassifier
import sfass as S
from grid import make_corr_data

RES = os.path.join(os.path.dirname(__file__), "..", "results")
CFG = dict(n=800, p=60, k=8, snr=1.4)
B_TREES = 150
CAP = 70
OUT = {}


def save():
    json.dump(OUT, open(os.path.join(RES, "camera.json"), "w"), indent=2, default=float)


def pearson(a, b): return float(np.corrcoef(a, b)[0, 1])
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
def msd(x): return (float(np.mean(x)), float(np.std(x)))


def fit_once(X, y, rng, B_trees=B_TREES, cap=CAP):
    """One paired-shadow forest fit -> (phi_real, phi_shadow, s_real, s_shadow)."""
    Xs, is_real = S.add_shadows(X, rng, paired=True)
    rf = RandomForestClassifier(n_estimators=B_trees, bootstrap=True,
                                max_features="sqrt", n_jobs=-1,
                                random_state=int(rng.integers(1 << 30)))
    rf.fit(Xs, y)
    pt, Nbi = S.per_tree_oob_importance(rf, Xs, y, rng, oob_cap=cap)
    phi, V_bc, V_raw, V_mc = S.ij_variance(pt, Nbi)
    s = np.sqrt(V_mc)
    return phi[is_real], phi[~is_real], s[is_real], s[~is_real]


def select(phi_r, phi_s, q):
    t = S.knockoff_threshold(phi_r, phi_s, q)
    return (phi_r >= t) if np.isfinite(t) else np.zeros(len(phi_r), bool)


# ---------------------------------------------------------------- H1
def phi_plain(X, y, rng, B_trees=B_TREES, cap=CAP, rs=None):
    """Plain forest (NO shadows): forest importance phi + between-tree s + IJ s."""
    rf = RandomForestClassifier(n_estimators=B_trees, bootstrap=True, n_jobs=-1,
                                max_features="sqrt",
                                random_state=rs if rs is not None else int(rng.integers(1 << 30)))
    rf.fit(X, y)
    pt, Nbi = S.per_tree_oob_importance(rf, X, y, rng, oob_cap=cap)
    phi, V_bc, V_raw, V_mc = S.ij_variance(pt, Nbi)
    return phi, np.sqrt(V_mc), np.sqrt(V_bc)


def exp_h1(R=25, seed=1):
    # Dispersion law: plain forests for BOTH ground truth and single fit (clean).
    print("[cam-H1] dispersion law + within-group + CI (plain forests)")
    n, p, k, snr = CFG["n"], CFG["p"], CFG["k"], CFG["snr"]
    rng = default_rng(seed)
    phis = np.zeros((R, p)); t0 = time.time()
    for r in range(R):
        Xr, yr, mask = make_corr_data(n, p, k, default_rng(1000 + r), snr=snr, rho=0.0)
        pr, _, _ = phi_plain(Xr, yr, rng)
        phis[r] = pr
        if r % 5 == 0: print(f"   gt {r}/{R} ({time.time()-t0:.0f}s)")
    true_sd = phis.std(axis=0, ddof=1)
    X0, y0, mask = make_corr_data(n, p, k, default_rng(1000), snr=snr, rho=0.0)
    # single-fit on X0 (same plain-forest recipe as the ground truth)
    rf = RandomForestClassifier(n_estimators=B_TREES, bootstrap=True, n_jobs=-1,
                                max_features="sqrt", random_state=7).fit(X0, y0)
    pt, Nbi = S.per_tree_oob_importance(rf, X0, y0, rng, oob_cap=CAP)
    phi0, V_bc, V_raw, V_mc = S.ij_variance(pt, Nbi)
    bt, ij = np.sqrt(V_mc), np.sqrt(V_bc)
    # bootstrap CI on the overall Pearson r (resample features)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, p, p)
        boot.append(pearson(bt[idx], true_sd[idx]))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    OUT["h1"] = dict(
        config=CFG, R=R,
        r_all=pearson(bt, true_sd), r_ci=ci,
        spearman_all=spearman(bt, true_sd),
        r_signal=pearson(bt[mask], true_sd[mask]),
        r_null=pearson(bt[~mask], true_sd[~mask]),
        spearman_signal=spearman(bt[mask], true_sd[mask]),
        spearman_null=spearman(bt[~mask], true_sd[~mask]),
        r_ij_all=pearson(ij, true_sd),
        r_ij_signal=pearson(ij[mask], true_sd[mask]),
        r_ij_null=pearson(ij[~mask], true_sd[~mask]),
        true_sd=true_sd.tolist(), bt=bt.tolist(), ij=ij.tolist(),
        mask=mask.tolist())
    print(f"   r_all={OUT['h1']['r_all']:.3f} CI={ci} | signal={OUT['h1']['r_signal']:.3f} "
          f"null={OUT['h1']['r_null']:.3f} | spearman={OUT['h1']['spearman_all']:.3f} "
          f"| IJ_all={OUT['h1']['r_ij_all']:.3f}")
    save()


# ---------------------------------------------------------------- E2/E3 + CPSS
def exp_main(seeds_sf=30, seeds_cpss=5, B_sub=50, q=0.2, seed=3):
    print("[cam-E2/E3] SF raw vs normalized (30 seeds) + CPSS (100 refits)")
    n, p, k, snr = CFG.values()
    raw = {"fdr": [], "pow": []}; norm = {"fdr": [], "pow": []}; sf_sec = []
    for r in range(seeds_sf):
        rng = default_rng(3000 + r)
        X, y, mask = make_corr_data(n, p, k, rng, snr=snr, rho=0.0)
        t = time.time()
        pr, ps_, sr, ss = fit_once(X, y, rng)
        sf_sec.append(time.time() - t)
        sel = select(pr, ps_, q); f, pw, _ = S.fdr_power(sel, mask)
        raw["fdr"].append(f); raw["pow"].append(pw)
        Wr, Ws = pr / (sr + 1e-12), ps_ / (ss + 1e-12)
        seln = select(Wr, Ws, q); fn, pwn, _ = S.fdr_power(seln, mask)
        norm["fdr"].append(fn); norm["pow"].append(pwn)
        if r % 10 == 0: print(f"   sf seed {r}")
    cp = {"fdr": [], "pow": []}; cp_sec = []
    for r in range(seeds_cpss):
        rng = default_rng(4000 + r)
        X, y, mask = make_corr_data(n, p, k, rng, snr=snr, rho=0.0)
        t = time.time()
        c = S.cpss(X, y, rng, B_sub=B_sub, B_trees=B_TREES, pi_thr=0.6)
        cp_sec.append(time.time() - t)
        f, pw, _ = S.fdr_power(c["sel"], mask); cp["fdr"].append(f); cp["pow"].append(pw)
        print(f"   cpss seed {r} ({c['n_fits']} fits, {time.time()-t:.0f}s)")
    ratio = np.array(cp_sec) / np.mean(sf_sec)
    OUT["main"] = dict(
        config=CFG, q=q, seeds_sf=seeds_sf, seeds_cpss=seeds_cpss,
        cpss_refits=2 * B_sub,
        raw_FDR=msd(raw["fdr"]), raw_power=msd(raw["pow"]),
        norm_FDR=msd(norm["fdr"]), norm_power=msd(norm["pow"]),
        cpss_FDR=msd(cp["fdr"]), cpss_power=msd(cp["pow"]),
        sf_sec=msd(sf_sec), cpss_sec=msd(cp_sec), speedup=msd(ratio))
    print("   RAW", OUT["main"]["raw_FDR"], OUT["main"]["raw_power"],
          "| CPSS", OUT["main"]["cpss_FDR"], OUT["main"]["cpss_power"],
          "| speedup", OUT["main"]["speedup"])
    save()


# ---------------------------------------------------------------- E5 label noise
def exp_e5(seeds=15, q=0.2, seed=6):
    print("[cam-E5] label-noise robustness (consistent config, 15 seeds)")
    n, p, k, snr = CFG.values()
    rows = []
    for ln in [0.0, 0.05, 0.10, 0.20]:
        fs, ps = [], []
        for r in range(seeds):
            rng = default_rng(6000 + r)
            X, y, mask = make_corr_data(n, p, k, rng, snr=snr, rho=0.0, label_noise=ln)
            pr, ps_, sr, ss = fit_once(X, y, rng)
            sel = select(pr, ps_, q); f, pw, _ = S.fdr_power(sel, mask)
            fs.append(f); ps.append(pw)
        rows.append(dict(noise=ln, FDR=msd(fs), power=msd(ps)))
        print(f"   noise={ln}: FDR={np.mean(fs):.3f} power={np.mean(ps):.3f}")
    OUT["e5"] = dict(config=CFG, rows=rows); save()


# ---------------------------------------------------------------- E6 grid
def exp_e6(seeds=30, seed=10):
    print("[cam-E6] FDR calibration + correlation (30 seeds)")
    n, p, k, snr = CFG.values()
    qs = [0.05, 0.10, 0.20]; out = {}
    for rho in [0.0, 0.3, 0.6, 0.9]:
        cell = {str(q): {"fdr": [], "pow": []} for q in qs}
        for r in range(seeds):
            rng = default_rng(10000 + r)
            X, y, mask = make_corr_data(n, p, k, rng, snr=snr, rho=rho)
            pr, ps_, sr, ss = fit_once(X, y, rng)
            for q in qs:
                sel = select(pr, ps_, q); f, pw, _ = S.fdr_power(sel, mask)
                cell[str(q)]["fdr"].append(f); cell[str(q)]["pow"].append(pw)
        out[str(rho)] = {q: dict(FDR=msd(cell[q]["fdr"]), power=msd(cell[q]["pow"])) for q in cell}
        print(f"   rho={rho}: q0.2 FDR={out[str(rho)]['0.2']['FDR']}")
        OUT["e6"] = dict(config=CFG, qs=qs, cells=out); save()


# ---------------------------------------------------------------- E9 forest size
def exp_e9(seeds=15, q=0.2, seed=20):
    print("[cam-E9] forest-size sensitivity")
    n, p, k, snr = CFG.values()
    rows = []
    for B in [50, 100, 200]:
        fs, ps, sec = [], [], []
        for r in range(seeds):
            rng = default_rng(20000 + r)
            X, y, mask = make_corr_data(n, p, k, rng, snr=snr, rho=0.0)
            t = time.time()
            pr, ps_, sr, ss = fit_once(X, y, rng, B_trees=B)
            sec.append(time.time() - t)
            sel = select(pr, ps_, q); f, pw, _ = S.fdr_power(sel, mask)
            fs.append(f); ps.append(pw)
        rows.append(dict(B=B, FDR=msd(fs), power=msd(ps), sec=msd(sec)))
        print(f"   B={B}: FDR={np.mean(fs):.3f} power={np.mean(ps):.3f} sec={np.mean(sec):.1f}")
    OUT["e9"] = dict(config=CFG, rows=rows); save()


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["h1", "e6", "e5", "e9", "main"]
    t0 = time.time()
    if os.path.exists(os.path.join(RES, "camera.json")):
        OUT.update(json.load(open(os.path.join(RES, "camera.json"))))
    for w in which:
        {"h1": exp_h1, "main": exp_main, "e5": exp_e5, "e6": exp_e6, "e9": exp_e9}[w]()
    print(f"[camera] done {time.time()-t0:.0f}s")
