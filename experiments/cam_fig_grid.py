"""Regenerate fig_grid.pdf from the unified-config camera.json E6 results."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
P = os.path.join(os.path.dirname(__file__), "..", "paper")
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3})
d = json.load(open(os.path.join(RES, "camera.json")))["e6"]
cells = d["cells"]; qs = [str(q) for q in d["qs"]]; qv = [float(q) for q in d["qs"]]
rhos = ["0.0", "0.3", "0.6", "0.9"]

fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.7))
for rho in rhos:
    fdr = [cells[rho][q]["FDR"][0] for q in qs]
    sd = [cells[rho][q]["FDR"][1] for q in qs]
    ax[0].errorbar(qv, fdr, yerr=sd, marker="o", capsize=2, label=f"$\\rho$={rho}")
ax[0].plot([0, 0.22], [0, 0.22], "k--", lw=1, label="target")
ax[0].set_xlabel("target $q$"); ax[0].set_ylabel("realized FDR")
ax[0].set_title("FDR calibration"); ax[0].legend(fontsize=6)

rv = [float(r) for r in rhos]
fdr = [cells[r]["0.2"]["FDR"][0] for r in rhos]
fsd = [cells[r]["0.2"]["FDR"][1] for r in rhos]
pw = [cells[r]["0.2"]["power"][0] for r in rhos]
ax[1].errorbar(rv, fdr, yerr=fsd, marker="o", capsize=2, color="C3", label="FDR")
ax[1].plot(rv, pw, "s-", color="C0", label="power")
ax[1].axhline(0.2, ls="--", c="k", lw=1, label="q=0.2")
ax[1].set_xlabel("equicorrelation $\\rho$"); ax[1].set_ylim(0, 1.05)
ax[1].set_title("vs. correlation ($q{=}0.2$)"); ax[1].legend(fontsize=6)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(P, f"fig_grid.{ext}"), bbox_inches="tight")
print("wrote fig_grid (unified config, 30 seeds, with error bars)")
