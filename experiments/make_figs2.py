"""E4 / E6 的论文插图"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
    "axes.linewidth": 0.8, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False,
})
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)
C = ["#1f3b73", "#b5432f", "#2e7d5b", "#8a5a00", "#6a3d9a"]

# ============================ Fig 4: E4 ============================
d4 = np.load(os.path.join(HERE, "e4_data.npz"))
j4 = json.load(open(os.path.join(HERE, "e4_result.json"), encoding="utf-8"))
ns = d4["N"].astype(float)
labels = [c["label"] for c in j4["cases"]]
pretty = ["EM, smooth", "averaging, smooth", "averaging, rough $\\gamma_s{=}0.42$",
          "averaging, rough + smoothed schedule", "averaging, aliased"]
mark = ["o-", "s--", "^-", "v-.", "D:"]

fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.75))

a = ax[0]
for i, (lab, pr, mk) in enumerate(zip(labels, pretty, mark)):
    y = d4[f"err_{i}"]
    a.loglog(ns, y, mk, color=C[i], ms=4, lw=1.15, label=pr)
a.set_xlabel(r"reverse sampler steps $N$")
a.set_ylabel(r"$\mathbb{E}\|X^{N}-X^{\mathrm{ref}}\|$")
a.set_title("(a) error vs. steps", fontsize=8.5)
a.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
a.legend(loc="lower left", fontsize=6.6)

b = ax[1]
gains = [j4["interventions"][l]["gain_8x_steps"] for l in labels]
bars = b.bar(range(len(labels)), gains, color=C, width=0.62, alpha=0.85)
b.axhline(8.0, ls="--", color="#b5432f", lw=1.1, label="ideal $8\\times$ (term C only)")
b.set_xticks(range(len(labels)))
b.set_xticklabels(["EM\nsmooth", "avg\nsmooth", "avg\nrough", "avg\nrough\n+sched", "avg\naliased"],
                  fontsize=7)
b.set_ylabel(r"error reduction for $8\times$ steps")
b.set_title("(b) return on extra steps", fontsize=8.5)
b.set_ylim(0, 9.4)
for r, g in zip(bars, gains):
    b.text(r.get_x() + r.get_width() / 2, g + 0.16, f"{g:.2f}", ha="center", fontsize=7)
b.legend(loc="lower left", fontsize=7)
b.grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_e4_intervention.pdf"))
fig.savefig(os.path.join(FIG, "fig_e4_intervention.png"), dpi=300)
plt.close(fig)

# ============================ Fig 5: E6 ============================
e6 = json.load(open(os.path.join(HERE, "e6_result.json"), encoding="utf-8"))
fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.75))

a = ax[0]
for i, tag in enumerate(["K1", "K8"]):
    hs = e6[tag]["delta_scan"]
    dd = np.array(sorted(float(k) for k in hs))
    yy = np.array([hs[str(x)] if str(x) in hs else hs[x] for x in dd])
    g = e6[tag]["gamma_s"]
    a.loglog(dd, yy, ["o-", "s-"][i], color=C[i], ms=4, lw=1.15,
             label=f"{'low' if tag=='K1' else 'high'}-freq time embed.: "
                   f"$\\gamma_s={g:.2f}$")
a.set_xlabel(r"time increment $\delta$")
a.set_ylabel(r"$\mathbb{E}\|s_\theta(x,t+\delta)-s_\theta(x,t)\|\,/\,\mathbb{E}\|s_\theta\|$")
a.set_title("(a) temporal Hölder regularity", fontsize=8.5)
a.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
a.legend(loc="upper left", fontsize=7)

b = ax[1]
for i, tag in enumerate(["K1", "K8"]):
    oc = e6[tag]["osc"]
    nn = np.array(sorted(int(k) for k in oc), float)
    yy = np.array([oc[str(int(n))]["mean_rel"] for n in nn])
    b.loglog(nn, yy, ["o-", "s-"][i], color=C[i], ms=4, lw=1.15,
             label=f"{'low' if tag=='K1' else 'high'}-freq: slope "
                   f"{e6[tag]['fitted_slope']:+.2f}")
    pred = e6[tag]["predicted_slope"]
    b.loglog(nn, yy[0] * (nn / nn[0]) ** pred, ":", color=C[i], lw=1.0)
b.set_xlabel(r"step count $N$")
b.set_ylabel(r"diagnostic $\mathrm{osc}(N)$")
b.set_title(r"(b) window-averaging residual ($\kappa=2/3$)", fontsize=8.5)
b.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
b.legend(loc="lower left", fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_e6_score_stats.pdf"))
fig.savefig(os.path.join(FIG, "fig_e6_score_stats.png"), dpi=300)
plt.close(fig)
print("saved fig_e4_intervention, fig_e6_score_stats")
