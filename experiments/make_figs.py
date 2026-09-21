"""Generate all paper figures from results/*.json into paper/figN.pdf+png."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
P = os.path.join(os.path.dirname(__file__), "..", "paper")
os.makedirs(P, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3})


def load(n):
    return json.load(open(os.path.join(R, n)))


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(P, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# Fig 1: H1 validity -- estimated SD vs true cross-resample SD
def fig_validity():
    d = load("h1_validity.json")
    true_sd = np.array(d["true_sd"]); bt = np.array(d["mc_sd"]); ij = np.array(d["ij_sd"])
    mask = np.array(d["mask"], dtype=bool)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for a, est, lab, c in [(ax[0], bt, "between-tree", "C0"),
                           (ax[1], ij, "infinitesimal jackknife", "C3")]:
        a.scatter(true_sd[~mask], est[~mask], s=14, c="0.6", label="null", alpha=.8)
        a.scatter(true_sd[mask], est[mask], s=22, c=c, label="signal", alpha=.9)
        r = np.corrcoef(true_sd, est)[0, 1]
        a.set_title(f"{lab}\nPearson r={r:.2f}")
        a.set_xlabel("true cross-resample SD of $\\phi_j$")
        a.legend(fontsize=7)
    ax[0].set_ylabel("single-fit SD estimate")
    fig.tight_layout(); savefig(fig, "fig_validity")


# Fig 2: H3 FDR/power vs CPSS + compute
def fig_fdr_compute():
    d = load("h3_fdr_compute.json")
    names = ["ours_raw", "ours_norm", "cpss"]
    labels = ["single-fit\n(raw)", "single-fit\n(norm.)", "CPSS\n(multi-fit)"]
    fdr = [d[n]["FDR"] for n in names]; pw = [d[n]["power"] for n in names]
    sec = [d[n]["sec_per_run"] for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    x = np.arange(3); w = .35
    ax[0].bar(x - w/2, fdr, w, label="FDR", color="C3")
    ax[0].bar(x + w/2, pw, w, label="power", color="C0")
    ax[0].axhline(d["q"], ls="--", c="k", lw=1, label=f"target q={d['q']}")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=7)
    ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=7); ax[0].set_title("FDR & power")
    ax[1].bar(x, sec, color=["C0", "C0", "C1"])
    ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=7)
    ax[1].set_ylabel("wall-clock (s, log)")
    ax[1].set_title(f"compute: {d['compute_ratio_wall']:.0f}$\\times$ speedup")
    fig.tight_layout(); savefig(fig, "fig_fdr_compute")


# Fig 3: scalability
def fig_scale():
    d = load("scale.json")["rows"]
    n = [r["n"] for r in d]; sf = [r["sfass_fit_s"] for r in d]
    cp = [r["cpss_est_s"] for r in d]
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.plot(n, sf, "o-", label="SF (1 fit)", color="C0")
    ax.plot(n, cp, "s--", label="CPSS (2$B$ fits)", color="C1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n samples (p=n/12.5)"); ax.set_ylabel("wall-clock (s)")
    ax.legend(fontsize=7); ax.set_title("scaling")
    fig.tight_layout(); savefig(fig, "fig_scale")


# Fig 4: E6 grid -- realized FDR vs q, and vs correlation
def fig_grid():
    d = load("grid.json")
    qs = [str(q) for q in d["qs"]]
    qv = [float(q) for q in d["qs"]]
    cs = d["corr_sweep"]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.7))
    # left: realized FDR vs target q, one line per rho
    for rho, cell in cs.items():
        fdr = [cell[q]["FDR"] for q in qs]
        sd = [cell[q]["FDR_sd"] for q in qs]
        ax[0].errorbar(qv, fdr, yerr=sd, marker="o", capsize=2, label=f"$\\rho$={rho}")
    ax[0].plot([0, 0.22], [0, 0.22], "k--", lw=1, label="target")
    ax[0].set_xlabel("target $q$"); ax[0].set_ylabel("realized FDR")
    ax[0].set_title("FDR calibration"); ax[0].legend(fontsize=6)
    # right: FDR & power vs rho at q=0.2
    rhos = sorted(cs.keys(), key=float)
    fdr = [cs[r]["0.2"]["FDR"] for r in rhos]
    fsd = [cs[r]["0.2"]["FDR_sd"] for r in rhos]
    pw = [cs[r]["0.2"]["power"] for r in rhos]
    ax[1].errorbar([float(r) for r in rhos], fdr, yerr=fsd, marker="o", capsize=2,
                   color="C3", label="FDR")
    ax[1].plot([float(r) for r in rhos], pw, "s-", color="C0", label="power")
    ax[1].axhline(0.2, ls="--", c="k", lw=1, label="q=0.2")
    ax[1].set_xlabel("equicorrelation $\\rho$"); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("vs. correlation ($q{=}0.2$)"); ax[1].legend(fontsize=6)
    fig.tight_layout(); savefig(fig, "fig_grid")


# Fig 5: E8 clustering restores FDR control under correlation
def fig_group():
    d = load("group.json")["synth_block"]
    rhos = sorted(d.keys(), key=float)
    x = [float(r) for r in rhos]
    pf = [d[r]["perfeat_FDR"] for r in rhos]
    gf = [d[r]["group_FDR"] for r in rhos]
    gfsd = [d[r].get("group_FDR_sd", 0) for r in rhos]
    gp = [d[r]["group_power"] for r in rhos]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.plot(x, pf, "s--", color="C3", label="per-feature FDR")
    ax.errorbar(x, gf, yerr=gfsd, marker="o", color="C0", capsize=2,
                label="group FDR (ours)")
    ax.plot(x, gp, "^:", color="C2", label="group power")
    ax.axhline(0.2, ls="--", c="k", lw=1, label="target q=0.2")
    ax.set_xlabel("block correlation $\\rho$"); ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate"); ax.set_title("Clustering restores FDR control")
    ax.legend(fontsize=6.5, loc="center left")
    fig.tight_layout(); savefig(fig, "fig_group")


if __name__ == "__main__":
    fig_validity()
    for fn in (fig_fdr_compute, fig_scale, fig_grid, fig_group):
        try:
            fn()
        except FileNotFoundError as e:
            print("skip (no data yet):", e)
