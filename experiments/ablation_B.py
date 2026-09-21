"""E9: sensitivity to forest size B (number of trees).
Does single-fit FDR/power stay stable as B varies, and what is the runtime cost?
Synthetic, rho=0 (low-correlation regime where per-feature selection applies).
Writes results/ablation_B.json.
"""
import os, json, time
import numpy as np
from numpy.random import default_rng
import sfass as S
from grid import make_corr_data

RES = os.path.join(os.path.dirname(__file__), "..", "results")

rows = []
for B in [50, 100, 200, 400]:
    fds, pws, secs = [], [], []
    for r in range(8):
        rng = default_rng(11000 + r)
        X, y, mask = make_corr_data(800, 50, 8, rng, snr=1.4, rho=0.0)
        t = time.time()
        rk = S.sf_ass_knockoff(X, y, rng, B_trees=B, q=0.2, variance="mc",
                               normalize=False, n_shadow=50, oob_cap=60)
        secs.append(time.time() - t)
        f, pw, _ = S.fdr_power(rk["sel"], mask); fds.append(f); pws.append(pw)
    rows.append(dict(B=B, FDR=float(np.mean(fds)), FDR_sd=float(np.std(fds)),
                     power=float(np.mean(pws)), sec=float(np.mean(secs))))
    print(f"B={B}: FDR={np.mean(fds):.3f} power={np.mean(pws):.3f} sec={np.mean(secs):.1f}")
    json.dump({"rows": rows}, open(os.path.join(RES, "ablation_B.json"), "w"), indent=2)
print("[E9] done")
