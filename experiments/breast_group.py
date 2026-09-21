"""Recompute the breast_cancer group-mode row with multiple shadow copies
(fixes the few-cluster BH floor), patch results/group.json."""
import os, json
import numpy as np
from numpy.random import default_rng
import group_select as G, sfass as S
from real_data import load_real

RES = os.path.join(os.path.dirname(__file__), "..", "results")
Xr, _ = load_real("breast_cancer"); p = Xr.shape[1]
gf, gp, pf, pp, ng = [], [], [], [], []
for r in range(10):
    rng = default_rng(95000 + r); rr = default_rng(5000 + r)
    signal = rr.choice(p, 6, replace=False)
    beta = np.zeros(p); beta[signal] = rr.uniform(0.8, 1.6, 6) * rr.choice([-1, 1], 6)
    y = (rr.random(Xr.shape[0]) < 1 / (1 + np.exp(-1.2 * (Xr @ beta)))).astype(int)
    mask = np.zeros(p, bool); mask[signal] = True
    sel, groups = G.group_select(Xr, y, rng, corr_thr=0.3, B_trees=120,
                                 oob_cap=60, n_shadow_copies=8)
    f, pw, ns, nr = G.eval_groups(sel, groups, mask)
    gf.append(f); gp.append(pw); ng.append(len(groups))
    rk = S.sf_ass_knockoff(Xr, y, rng, B_trees=120, q=0.2, variance="mc",
                           normalize=False, n_shadow=p, oob_cap=60)
    pfd, ppw, _ = S.fdr_power(rk["sel"], mask); pf.append(pfd); pp.append(ppw)
    print(f"rep {r}: group FDR={f:.2f} pow={pw:.2f} | perfeat FDR={pfd:.2f} pow={ppw:.2f}")

d = json.load(open(os.path.join(RES, "group.json")))
d["breast"] = dict(group_FDR=float(np.mean(gf)), group_FDR_sd=float(np.std(gf)),
                   group_power=float(np.mean(gp)), n_groups=float(np.mean(ng)),
                   n_shadow_copies=8,
                   perfeat_FDR=float(np.mean(pf)), perfeat_power=float(np.mean(pp)))
json.dump(d, open(os.path.join(RES, "group.json"), "w"), indent=2)
print("PATCHED breast:", d["breast"])
