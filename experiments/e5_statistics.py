"""
E5: 所有拟合指数的 bootstrap 置信区间与假设检验

方法
----
对每一档 N，从该档的逐路径误差中有放回重采样（B=2000），
重算均值，再对 (log N, log err) 做加权最小二乘，得到斜率的 bootstrap 分布。
报告 95% 百分位区间，并以双侧检验给出与理论预测值的一致性判断。

输出
----
e5_result.json       —— 所有斜率、CI、p 值
tab_stats.tex        —— 可直接 \input 进论文的表
"""

import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = 2000
RNG = np.random.default_rng(20261101)


def boot_slope(ns, per_path, B=B, rng=RNG):
    """per_path: list of arrays, len = len(ns)"""
    ns = np.asarray(ns, float)
    slopes = np.empty(B)
    n = len(ns)
    for b in range(B):
        ys = np.empty(n)
        for i, arr in enumerate(per_path):
            ys[i] = rng.choice(arr, size=len(arr), replace=True).mean()
        slopes[b] = np.polyfit(np.log(ns), np.log(ys), 1)[0]
    return slopes


def summarize(slopes, point, h0=None, name=""):
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    se = slopes.std(ddof=1)
    out = {"name": name, "slope": float(point), "ci95": [float(lo), float(hi)],
           "se": float(se), "n_boot": int(len(slopes))}
    if h0 is not None:
        # 双侧 percentile p 值：bootstrap 分布中越过 h0 的比例
        frac = float((slopes > h0).mean())
        p = 2 * min(frac, 1 - frac)
        out["h0"] = float(h0)
        out["p_value"] = float(min(1.0, p))
        out["consistent_with_h0"] = bool(lo <= h0 <= hi)
    return out


def main():
    res = {}

    # ---------------- E1 ----------------
    d1 = np.load(os.path.join(HERE, "e1_data.npz"))
    ns = d1["N"]
    pp = [d1[f"paths_{int(n)}"] for n in ns]
    point = np.polyfit(np.log(ns.astype(float)), np.log(np.array([a.mean() for a in pp])), 1)[0]
    sl = boot_slope(ns, pp)
    res["E1"] = summarize(sl, point, h0=-0.25, name="E1 forward rate (gamma=0.25)")
    res["E1"]["theory_bound"] = -0.25
    res["E1"]["classical"] = -0.5

    # ---------------- E3a ----------------
    d3 = np.load(os.path.join(HERE, "e3_data.npz"))
    ns3 = d3["N"]
    pp3 = [d3[f"paths_{int(n)}"] for n in ns3]
    point3 = np.polyfit(np.log(ns3.astype(float)), np.log(np.array([a.mean() for a in pp3])), 1)[0]
    sl3 = boot_slope(ns3, pp3)
    res["E3a"] = summarize(sl3, point3, h0=-0.5, name="E3a discretisation term (C)")

    # ---------------- E3b ----------------
    dd = d3["delta"]
    ppd = [d3[f"dpaths_{k}"] for k in dd]
    means = np.array([a.mean() for a in ppd])
    slopes_b = np.empty(B)
    for b in range(B):
        ys = np.array([RNG.choice(a, size=len(a), replace=True).mean() for a in ppd])
        slopes_b[b] = np.polyfit(dd[1:], ys[1:], 1)[0]
    pt_b = np.polyfit(dd[1:], means[1:], 1)[0]
    res["E3b"] = summarize(slopes_b, pt_b, h0=0.0, name="E3b score term (B) slope")

    # ---------------- E4 ----------------
    d4 = np.load(os.path.join(HERE, "e4_data.npz"))
    ns4 = d4["N"]
    e4j = json.load(open(os.path.join(HERE, "e4_result.json"), encoding="utf-8"))
    labels = [c["label"] for c in e4j["cases"]]
    res["E4"] = {}
    for i, lab in enumerate(labels):
        pp4 = [d4[f"c{i}_N{int(n)}"] for n in ns4]
        pt = np.polyfit(np.log(ns4.astype(float)), np.log(np.array([a.mean() for a in pp4])), 1)[0]
        sl4 = boot_slope(ns4, pp4)
        res["E4"][lab] = summarize(sl4, pt, h0=-1.0, name=f"E4 {lab}")
        # 8x 步数收益的 bootstrap
        gains = np.empty(B)
        for b in range(B):
            g32 = RNG.choice(pp4[1], size=len(pp4[1]), replace=True).mean()
            g256 = RNG.choice(pp4[4], size=len(pp4[4]), replace=True).mean()
            gains[b] = g32 / g256
        lo, hi = np.percentile(gains, [2.5, 97.5])
        res["E4"][lab]["gain_8x"] = {
            "point": float(np.array([a.mean() for a in pp4])[1] /
                           np.array([a.mean() for a in pp4])[4]),
            "ci95": [float(lo), float(hi)],
        }

    # ---------------- E6 ----------------
    e6 = json.load(open(os.path.join(HERE, "e6_result.json"), encoding="utf-8"))
    res["E6"] = {}
    for tag in ("K1", "K8"):
        r = e6[tag]
        res["E6"][tag] = {
            "gamma_s": r["gamma_s"],
            "osc_slope": r["fitted_slope"],
            "predicted": r["predicted_slope"],
            "final_dsm_loss": r["final_loss"],
        }

    # ---------------- 打印 ----------------
    print("=" * 78)
    print(f"{'quantity':38s}{'slope':>10s}{'95% CI':>22s}{'H0':>8s}{'p':>9s}")
    print("=" * 78)

    def line(d):
        ci = f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
        h0 = f"{d['h0']:+.2f}" if "h0" in d else "-"
        p = f"{d['p_value']:.3f}" if "p_value" in d else "-"
        print(f"{d['name'][:37]:38s}{d['slope']:>10.3f}{ci:>22s}{h0:>8s}{p:>9s}")

    line(res["E1"]); line(res["E3a"]); line(res["E3b"])
    for lab in labels:
        line(res["E4"][lab])
    print("-" * 78)
    for tag in ("K1", "K8"):
        r = res["E6"][tag]
        print(f"E6 {tag}: gamma_s={r['gamma_s']:.3f}  osc slope={r['osc_slope']:+.3f} "
              f"(predicted {r['predicted']:+.3f})")

    with open(os.path.join(HERE, "e5_result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # ---------------- LaTeX 表 ----------------
    rows = []
    rows.append(("E1 forward, trajectory rate", res["E1"], r"$-(1/2-\gamma)$"))
    rows.append(("E3a discretisation (C)", res["E3a"], r"$-1/2$"))
    rows.append(("E3b score term (B)", res["E3b"], r"linear, slope $>0$"))
    for lab in labels:
        rows.append((f"E4 {lab}", res["E4"][lab], r"$-1$"))
    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Fitted exponents with bootstrap confidence intervals "
               r"($B=2000$ resamples over trajectories) and two-sided tests against "
               r"the predicted exponent.}")
    tex.append(r"\label{tab:stats}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{@{}lccc@{}}")
    tex.append(r"\toprule")
    tex.append(r"quantity & fitted exponent & 95\% CI & $p$ vs.\ prediction \\")
    tex.append(r"\midrule")
    for name, d, pred in rows:
        ci = f"[{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
        p = f"{d['p_value']:.3f}" if "p_value" in d else "--"
        star = "" if (d.get("p_value", 1.0) > 0.05) else "$^{*}$"
        tex.append(f"{name} & {d['slope']:+.3f} & {ci} & {p}{star} \\\\")
    tex.append(r"\midrule")
    for tag in ("K1", "K8"):
        r = res["E6"][tag]
        tex.append(f"E6 {tag}: $\\gamma_s$ & {r['gamma_s']:.3f} & -- & "
                   f"osc slope {r['osc_slope']:+.3f} vs.\\ {r['predicted']:+.3f} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    with open(os.path.join(HERE, "tab_stats.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    print("\n[E5] saved e5_result.json / tab_stats.tex")


if __name__ == "__main__":
    main()
