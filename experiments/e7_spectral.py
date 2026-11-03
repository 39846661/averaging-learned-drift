"""E7: direct verification of the spectral (Nyquist) criterion, Proposition 4.4.

For a pure temporal tone s(x,t) = sbar(x) + A sin(2 pi f t) the window residual of
Hypothesis H is exactly A |D_M(2 pi f / N)|, with D_M the Dirichlet kernel of the
M-point window. This script compares that closed form with a direct evaluation of
the window average on a dense grid of start times, for frequencies that are slow,
intermediate, resonant and grid-aliased.

Outputs: e7_result.json, tab_spectral.tex
"""

import json
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))


def dirichlet(theta, M):
    """D_M(theta) = (1/M) sum_{j<M} e^{i j theta}, stable closed form."""
    theta = np.asarray(theta, dtype=float)
    s = np.sin(0.5 * theta)
    # limit value 1 at theta in 2 pi Z
    num = np.sin(0.5 * M * theta)
    den = M * s
    out = np.where(np.abs(s) < 1e-14, 1.0, num / np.where(den == 0, 1.0, den))
    return np.abs(out)


def measured_residual(f, N, M, n_t=4001):
    """sup_t |(1/M) sum_j sin(2 pi f (t + j/N))|, amplitude A = 1."""
    t = np.linspace(0.0, 1.0, n_t)
    j = np.arange(M)
    phase = 2.0 * np.pi * f * (t[:, None] + j[None, :] / N)
    return float(np.abs(np.sin(phase).mean(axis=1)).max())


def delta_N(f, N):
    """distance from f to the nearest multiple of N."""
    m = round(f / N)
    return abs(f - m * N)


def main():
    N = 256
    M = int(np.floor(N ** (2.0 / 3.0)))  # kappa = 2/3, matching E4 and E6
    band = N / (3.0 * M)

    tones = [
        (1, "slow, inside the profile"),
        (30, "below the window resolution"),
        (128, "half the sampling frequency"),
        (250, "just outside the resonant band"),
        (256, "grid resonance, f = N"),
        (257, "inside the resonant band"),
        (65536, "aliasing counterexample, f = N^2"),
    ]

    rows = []
    for f, note in tones:
        pred = float(dirichlet(2.0 * np.pi * f / N, M))
        meas = measured_residual(f, N, M)
        rows.append(
            dict(
                f=f,
                note=note,
                delta=float(delta_N(f, N)),
                predicted=pred,
                measured=meas,
                regime=("in band" if delta_N(f, N) <= band else "averaged"),
            )
        )

    result = dict(N=N, M=M, band_halfwidth=band, rows=rows)
    with open(os.path.join(OUT, "e7_result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    # LaTeX table
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{E7: the spectral criterion of \Cref{prop:spectral} checked directly."
        r" Window average of a unit-amplitude tone at frequency $f$, with $N=%d$ and"
        r" $M=\lfloor N^{2/3}\rfloor=%d$ (resonant band half-width $N/(3M)=%.2f$)."
        r" Predicted value is $|\mathcal{D}_{M}(2\pi f/N)|$ from \eqref{eq:tone-residual};"
        r" measured value is the supremum over $4001$ start times.}" % (N, M, band),
        r"\label{tab:spectral}",
        r"\begin{tabular}{@{}rrlrr@{}}",
        r"\toprule",
        r"$f$ & $\delta_{N}(f)$ & regime & predicted & measured \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            r"%d & %g & %s & %.4f & %.4f \\" % (r["f"], r["delta"], r["regime"], r["predicted"], r["measured"])
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    with open(os.path.join(OUT, "tab_spectral.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    for r in rows:
        print(f"f={r['f']:>6}  delta={r['delta']:>6.0f}  {r['regime']:>9}  pred={r['predicted']:.4f}  meas={r['measured']:.4f}")
    print(f"\nN={N}, M={M}, band half-width={band:.3f}")


if __name__ == "__main__":
    main()
