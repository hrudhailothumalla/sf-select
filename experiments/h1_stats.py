"""Within-group correlations, Spearman, and bootstrap CI for the dispersion law,
computed from the existing clean plain-forest run (results/h1_validity.json).
No model fitting -- pure re-analysis. Addresses reviewer R1.3 / R2.2."""
import os, json
import numpy as np
from numpy.random import default_rng

RES = os.path.join(os.path.dirname(__file__), "..", "results")
d = json.load(open(os.path.join(RES, "h1_validity.json")))
true = np.array(d["true_sd"]); bt = np.array(d["mc_sd"]); ij = np.array(d["ij_sd"])
mask = np.array(d["mask"], dtype=bool)
p = len(true); rng = default_rng(0)


def pear(a, b): return float(np.corrcoef(a, b)[0, 1])
def spear(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def boot_ci(a, b, n=5000):
    rs = []
    for _ in range(n):
        idx = rng.integers(0, len(a), len(a))
        rs.append(pear(a[idx], b[idx]))
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


out = dict(
    p=p, n_signal=int(mask.sum()), n_null=int((~mask).sum()),
    bt_r_all=pear(bt, true), bt_r_ci=boot_ci(bt, true),
    bt_r_signal=pear(bt[mask], true[mask]), bt_r_null=pear(bt[~mask], true[~mask]),
    bt_sp_all=spear(bt, true), bt_sp_signal=spear(bt[mask], true[mask]),
    bt_sp_null=spear(bt[~mask], true[~mask]),
    ij_r_all=pear(ij, true), ij_r_signal=pear(ij[mask], true[mask]),
    ij_r_null=pear(ij[~mask], true[~mask]),
)
json.dump(out, open(os.path.join(RES, "h1_within.json"), "w"), indent=2)
for k, v in out.items():
    print(f"{k:14s}: {v}")
