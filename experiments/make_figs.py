"""把 E1/E2/E3 的结果画成论文用的矢量图（PDF + PNG）"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

C_MAIN = "#1f3b73"
C_ALT = "#b5432f"
C_GREY = "#7a7a7a"


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- Fig 1: E1
e1 = load("e1_result.json")
N = np.array(e1["N_list"], dtype=float)
y = np.array(e1["error_curves"]["0.25"])
slope = e1["fitted"]["0.25"]["slope"]

fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.loglog(N, y, "o-", color=C_MAIN, ms=4, lw=1.2,
          label=f"measured ($N^{{{slope:.2f}}}$)")
# 理论下界 N^{-(1/2-gamma)}，锚定在 N=16 处
anchor = y[0]
ax.loglog(N, anchor * (N / N[0]) ** (-0.25), "--", color=C_ALT, lw=1.1,
          label=r"theorem bound $N^{-(1/2-\gamma)}$")
ax.loglog(N, anchor * (N / N[0]) ** (-0.5), ":", color=C_GREY, lw=1.0,
          label=r"classical EM $N^{-1/2}$")
ax.set_xlabel(r"number of steps $N$")
ax.set_ylabel(r"$\mathbb{E}\,\|X^{N}-\bar X\|_{\gamma}$")
ax.set_title(r"(a) Forward scheme: trajectory-level rate ($\gamma=0.25$)", fontsize=8.5)
ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_e1_rate.pdf"))
fig.savefig(os.path.join(FIG, "fig_e1_rate.png"), dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Fig 2: E2
e2 = load("e2_result.json")
N2 = np.array(e2["N_list"], dtype=float)
sup = np.array(e2["sup_error"])
s2 = e2["fitted_slope_sup"]

fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.loglog(N2, sup, "s-", color=C_MAIN, ms=3.5, lw=1.2,
          label=f"measured ($N^{{{s2:.2f}}}$)")
ax.loglog(N2, sup[0] * (N2 / N2[0]) ** (-1.0), "--", color=C_ALT, lw=1.1,
          label=r"$N^{-1}$")
ax.loglog(N2, sup[0] * (N2 / N2[0]) ** (-2.0), ":", color=C_GREY, lw=1.0,
          label=r"$N^{-2}$")
ax.set_xlabel(r"number of chain steps $N$")
ax.set_ylabel(r"$\sup_k |X_k - Y_k|$")
ax.set_title("(b) Embedding remainder: per-step $O(N^{-2})$, accumulated $O(N^{-1})$",
             fontsize=8.5)
ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_e2_embedding.pdf"))
fig.savefig(os.path.join(FIG, "fig_e2_embedding.png"), dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Fig 3: E3
e3 = load("e3_result.json")
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))

# (a) dt 扫描
Nd = np.array(e3["E3a_dt_scan"]["N"], dtype=float)
ed = np.array(e3["E3a_dt_scan"]["err"])
sa = e3["E3a_dt_scan"]["fitted_slope"]
ax = axes[0]
ax.loglog(Nd, ed, "o-", color=C_MAIN, ms=4, lw=1.2,
          label=f"measured ($N^{{{sa:.2f}}}$)")
ax.loglog(Nd, ed[0] * (Nd / Nd[0]) ** (-1.0), "--", color=C_ALT, lw=1.1,
          label=r"$N^{-1}$")
ax.loglog(Nd, ed[0] * (Nd / Nd[0]) ** (-0.5), ":", color=C_GREY, lw=1.0,
          label=r"$N^{-1/2}$")
ax.set_xlabel(r"reverse sampler steps $N$")
ax.set_ylabel(r"$\mathbb{E}\|X^{N}-X^{\mathrm{ref}}\|$")
ax.set_title(r"(a) term (C): discretisation, $\delta_\theta=0$", fontsize=8.5)
ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
ax.legend(loc="lower left")

# (b) delta 扫描
dd = np.array(e3["E3b_delta_scan"]["delta"])
ed2 = np.array(e3["E3b_delta_scan"]["err"])
slb = e3["E3b_delta_scan"]["linear_slope"]
icb = e3["E3b_delta_scan"]["linear_intercept"]
ax = axes[1]
ax.plot(dd, ed2, "o", color=C_MAIN, ms=4, label="measured")
xx = np.linspace(0, dd.max(), 100)
ax.plot(xx, slb * xx + icb, "--", color=C_ALT, lw=1.1,
        label=rf"fit: ${slb:.2f}\,\delta_\theta + {icb:.3f}$")
ax.set_xlabel(r"score error $\delta_\theta$")
ax.set_ylabel(r"$\mathbb{E}\|X^{N}-X^{\mathrm{ref}}\|$")
ax.set_title(r"(b) term (B): score error, $N=512$", fontsize=8.5)
ax.grid(True, ls=":", lw=0.4, alpha=0.5)
ax.legend(loc="upper left")

fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_e3_decomposition.pdf"))
fig.savefig(os.path.join(FIG, "fig_e3_decomposition.png"), dpi=300)
plt.close(fig)

print("figures written to", FIG)
for f in sorted(os.listdir(FIG)):
    print("  ", f)
