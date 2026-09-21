"""
Follow-ups demanded by the correlation finding:
 (A) Fair comparison UNDER correlation (rho=0.3): does CPSS also lose FDR
     control, or is SF-Select uniquely fragile?  -> decides how to qualify
     the "beats CPSS" claim.
 (B) SF-Select FDR across sizes AT independence (rho=0): show control holds
     across (n,p) when features are ~independent.
Writes results/followup.json.
"""
import os, json, time
import numpy as np
from numpy.random import default_rng
import sfass as S
from grid import make_corr_data, fit_stats, eval_at_q

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def run():
    out = {}
    t0 = time.time()

    # (A) SF vs CPSS at rho=0.3
    print("[FU-A] SF vs CPSS under correlation rho=0.3 (10 reps)")
    sf, cp = {"fdr": [], "pow": []}, {"fdr": [], "pow": []}
    for r in range(10):
        rng = default_rng(70000 + r)
        X, y, mask = make_corr_data(800, 60, 8, rng, snr=1.4, rho=0.3)
        Wr, Ws = fit_stats(X, y, rng, B_trees=100, cap=55)
        f, pw, _ = eval_at_q(Wr, Ws, mask, 0.2)
        sf["fdr"].append(f); sf["pow"].append(pw)
        c = S.cpss(X, y, rng, B_sub=8, B_trees=100, pi_thr=0.6)
        cf, cpw, _ = S.fdr_power(c["sel"], mask)
        cp["fdr"].append(cf); cp["pow"].append(cpw)
        print(f"   rep {r}: SF FDR={f:.2f} pow={pw:.2f} | CPSS FDR={cf:.2f} pow={cpw:.2f}"
              f"  ({time.time()-t0:.0f}s)")
    out["corr_compare_rho0.3"] = dict(
        sf_FDR=float(np.mean(sf["fdr"])), sf_FDR_sd=float(np.std(sf["fdr"])),
        sf_power=float(np.mean(sf["pow"])),
        cpss_FDR=float(np.mean(cp["fdr"])), cpss_FDR_sd=float(np.std(cp["fdr"])),
        cpss_power=float(np.mean(cp["pow"])))
    json.dump(out, open(os.path.join(RES, "followup.json"), "w"), indent=2)
    print("   ->", out["corr_compare_rho0.3"])

    # (B) SF size sweep at rho=0
    print("[FU-B] SF FDR across sizes at independence rho=0 (20 reps, q=0.2)")
    out["size_indep"] = {}
    for (n, p, k) in [(500, 40, 6), (1000, 100, 10), (2000, 200, 15)]:
        fdrs, pws = [], []
        for r in range(20):
            rng = default_rng(80000 + r)
            X, y, mask = make_corr_data(n, p, k, rng, snr=1.4, rho=0.0)
            Wr, Ws = fit_stats(X, y, rng, B_trees=100, cap=55)
            f, pw, _ = eval_at_q(Wr, Ws, mask, 0.2)
            fdrs.append(f); pws.append(pw)
        out["size_indep"][f"n{n}_p{p}"] = dict(
            FDR=float(np.mean(fdrs)), FDR_sd=float(np.std(fdrs)),
            power=float(np.mean(pws)))
        print(f"   n={n} p={p}: FDR={np.mean(fdrs):.3f}+-{np.std(fdrs):.3f} "
              f"power={np.mean(pws):.3f}  ({time.time()-t0:.0f}s)")
        json.dump(out, open(os.path.join(RES, "followup.json"), "w"), indent=2)
    print(f"[FU] done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
