"""
E4: 推论（步长与调度之别）的干预实验

被检验的判据
------------
(C) 项主导区：减小 Delta t（加步数 N）显著改善；改变调度形状几乎无效。
(A) 项主导区：加步数收益递减（进入 A 项地板）；改变调度形状显著改善。

设计的关键
----------
三种调度的**积分完全相同**：int_0^1 beta(t) dt = 5.05，因此
    abar_T = exp(-5.05) ~ 6.4e-3
三者相同，(D) 项（有限 T 截断）完全一致，不构成混淆。三者只在 beta 的
**时间形状**上不同：

    const   beta(t) = 5.05                                   (beta' = 0)
    linear  beta(t) = 0.1 + 9.9 t                            (beta' = 9.9)
    oscil   beta(t) = 5.05 (1 + 0.9 sin(24 pi t))            (beta' 最大 ~ 342)

score 对 t 的依赖经由 abar(t) 进入，所以 beta 的时间结构直接控制 score 的
时间 Holder 常数与指数：beta 变化越快，窗口平均 omega_N 越大，A 项地板越高。

这也是 E6 的机制在可控系统上的对照实现：真实网络上测到 gamma_s = 0.47 < 1，
说明 omega_N 只以 N^{-gamma_s(1-kappa)} 衰减，因此 A 项迟早接管。
"""

import os
import json
import numpy as np

RNG = np.random.default_rng(20261101)
MU = np.array([[-1.5, 0.0], [1.5, 0.0]])
W = np.array([0.5, 0.5])
D = 2
T_MAX = 1.0
TOTAL_BETA = 5.05
N_REF = 2048


# ---------------------------------------------------------------- 调度
def beta_of(kind, t):
    if kind == "const":
        return np.full_like(t, TOTAL_BETA)
    if kind == "linear":
        return 0.1 + 9.9 * t
    if kind == "oscil":
        return TOTAL_BETA * (1.0 + 0.9 * np.sin(24.0 * np.pi * t))
    raise ValueError(kind)


_ABAR_CACHE = {}


def abar_of(kind, t):
    """数值积分 abar(t) = exp(-int_0^t beta)，预先建表插值"""
    if kind not in _ABAR_CACHE:
        grid = np.linspace(0.0, T_MAX, 20001)
        b = beta_of(kind, grid)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (b[1:] + b[:-1]) * np.diff(grid))])
        _ABAR_CACHE[kind] = (grid, np.exp(-cum))
    g, a = _ABAR_CACHE[kind]
    return np.interp(t, g, a)


_WEI_CACHE = {}


def weierstrass(t, gamma, K=11):
    """W(t) = sum_k a^k cos(2^k pi t)，a = 2^{-gamma}，Hölder 指数恰为 gamma"""
    key = (round(float(gamma), 4), K)
    if key not in _WEI_CACHE:
        a = 2.0 ** (-gamma)
        ks = np.arange(K)
        _WEI_CACHE[key] = a ** ks
    a_pow = _WEI_CACHE[key]
    ks = np.arange(K)
    z = t[:, None] * (2.0 ** ks)[None, :] * np.pi
    w = (np.cos(z) * a_pow[None, :]).sum(axis=1)
    return w / (a_pow.sum() + 1e-12)      # 归一到 |W| <= 1


def score_true(kind, x, t, rough_gamma=None, rough_amp=0.0, alias_amp=0.0, alias_N=0):
    a = abar_of(kind, t)
    m = np.sqrt(a)[:, None, None] * MU[None, :, :]     # (M, 2, 2)
    d2 = ((x[:, None, :] - m) ** 2).sum(axis=2)        # (M, 2)
    lp = np.log(W)[None, :] - 0.5 * d2
    lp -= lp.max(axis=1, keepdims=True)
    wgt = np.exp(lp)
    wgt /= wgt.sum(axis=1, keepdims=True)
    s = -x + np.sqrt(a)[:, None] * (wgt @ MU)
    if rough_amp > 0 and rough_gamma is not None:
        # 加一个幅度为 rough_amp、Hölder 指数为 rough_gamma 的时间涨落。
        # 方向取归一化的 x（切向），保证扰动的是 score 场而非仅平移。
        v = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        mod = (beta_of(kind, t) / TOTAL_BETA)[:, None]
        s = s + rough_amp * mod * weierstrass(t, rough_gamma)[:, None] * v
    if alias_amp > 0 and alias_N > 0:
        # Prop. 反例的实现：sin(2 pi (N^2+1) t)。网格点 t+j/N 上相位为 2 pi (k+j)/N，
        # 随 j 变化但平均后不趋于 g(t)，故窗口平均残差不随 N 消失
        v = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        s = s + alias_amp * np.sin(2.0 * np.pi * (alias_N ** 2 + 1) * t)[:, None] * v
    return s


def sample_xT(kind, n):
    idx = RNG.choice(2, size=n, p=W)
    aT = float(abar_of(kind, np.array([T_MAX]))[0])
    return np.sqrt(aT) * MU[idx] + RNG.standard_normal((n, D))


# ---------------------------------------------------------------- 采样
def reverse_sde(kind, xT, n_steps, xi, rg=None, ra=0.0, avg_M=0, alias_amp=0.0):
    """avg_M=0 为标准 Euler-Maruyama（点值 score）；
    avg_M>0 为平均化采样器：每步的 score 取 M 点窗口平均，
    即定理中真正与平均化极限相比较的那个过程。"""
    x = xT.copy()
    ts = np.linspace(T_MAX, 0.0, n_steps + 1)
    for i in range(n_steps):
        t = np.full(len(x), ts[i])
        h = abs(ts[i + 1] - ts[i])
        b = beta_of(kind, t)
        if avg_M > 0:
            acc = np.zeros_like(x)
            for j in range(avg_M):
                tj = np.full(len(x), min(ts[i] + j / n_steps, T_MAX))
                acc += score_true(kind, x, tj, rg, ra, alias_amp, n_steps)
            s = acc / avg_M
        else:
            s = score_true(kind, x, t, rg, ra, alias_amp, n_steps)
        x = x - (0.5 * b[:, None] * x + b[:, None] * s) * h + np.sqrt(b * h)[:, None] * xi[:, i, None]
    return x


def osc_of(kind, n_steps, kappa=2.0 / 3.0, n_x=400, n_t=10, t_lo=0.10, t_hi=0.80, rg=None, ra=0.0):
    """诊断量 osc：窗口平均 score 的相对偏差"""
    M = max(2, int(np.floor(n_steps ** kappa)))
    xs = MU[RNG.choice(2, size=n_x, p=W)] + RNG.standard_normal((n_x, D))
    ts = np.linspace(t_lo, t_hi, n_t)
    rel = []
    for t0 in ts:
        js = np.arange(M) / n_steps
        if t0 + js[-1] > T_MAX:
            continue
        acc = np.zeros_like(xs)
        for j in js:
            acc += score_true(kind, xs, np.full(n_x, t0 + j), rg, ra)
        s0 = score_true(kind, xs, np.full(n_x, t0), rg, ra)
        num = np.linalg.norm(acc / M - s0, axis=1)
        den = np.linalg.norm(s0, axis=1) + 1e-8
        rel.extend((num / den).tolist())
    return float(np.mean(rel))


# ---------------------------------------------------------------- 主实验
def run_case(label, kind, rg, ra, use_avg, N_LIST, M=200, kappa=2.0 / 3.0, alias=0.0):
    """跑一个配置：配对参考解 + 逐档 N"""
    xT = sample_xT(kind, M)
    xi_ref = RNG.standard_normal((M, N_REF))
    if use_avg:
        xref = reverse_sde(kind, xT, N_REF, xi_ref, rg=rg, ra=ra,
                           avg_M=max(2, int(N_REF ** kappa)), alias_amp=alias)
    else:
        xref = reverse_sde(kind, xT, N_REF, xi_ref, rg=rg, ra=ra, alias_amp=alias)
    errs = []
    per_path = {}
    for N in N_LIST:
        blk = N_REF // N
        xi_c = xi_ref[:, : blk * N].reshape(M, N, blk).sum(axis=2) / np.sqrt(blk)
        mv = max(2, int(N ** kappa)) if use_avg else 0
        xe = reverse_sde(kind, xT, N, xi_c, rg=rg, ra=ra, avg_M=mv, alias_amp=alias)
        ei = np.linalg.norm(xe - xref, axis=1)
        per_path[N] = ei
        e = float(ei.mean())
        errs.append(e)
        print(f"    N={N:5d}  err={e:.5e}" + (f"  (M={mv})" if use_avg else ""), flush=True)
    ns = np.array(N_LIST, dtype=float)
    s_all, _ = np.polyfit(np.log(ns), np.log(errs), 1)
    s_tail, _ = np.polyfit(np.log(ns[-3:]), np.log(errs[-3:]), 1)
    print(f"    -> slope overall {s_all:+.3f} | tail {s_tail:+.3f}")
    return {"label": label, "kind": kind, "gamma": rg, "amp": ra,
            "avg": bool(use_avg), "alias": alias, "N": N_LIST, "err": errs,
            "per_path": per_path,
            "slope_all": float(s_all), "slope_tail": float(s_tail)}


def main():
    N_LIST = [16, 32, 64, 128, 256]
    here = os.path.dirname(os.path.abspath(__file__))
    cases = []
    AMP = 0.8

    print("=" * 70)
    print("E4: which intervention works, and in which regime")
    print("=" * 70)

    print("\n[1] standard EM, smooth score  (C-dominated reference)")
    cases.append(run_case("EM smooth", "const", None, 0.0, False, N_LIST))

    print("\n[2] averaging sampler, smooth score  (term A present but tiny)")
    cases.append(run_case("AVG smooth", "const", None, 0.0, True, N_LIST))

    print(f"\n[3] averaging sampler, rough score gamma=0.42 amp={AMP}  (A-dominated)")
    cases.append(run_case("AVG rough", "const", 0.42, AMP, True, N_LIST))

    print(f"\n[4] averaging sampler, rough score, SMOOTHED schedule  (schedule fix)")
    cases.append(run_case("AVG rough+smooth-sched", "const", 0.42, AMP * 0.25, True, N_LIST))

    print("\n[5] averaging sampler, ALIASED score sin(2 pi N^2 t) (Prop. counterexample)")
    cases.append(run_case("AVG aliased", "const", None, 0.0, True, N_LIST, alias=0.8))

    # ---------------- 干预矩阵 ----------------
    print("\n" + "=" * 70)
    print("Intervention matrix")
    print("=" * 70)
    table = {}
    for c in cases:
        e = np.array(c["err"])
        table[c["label"]] = {
            "gain_8x_steps": float(e[1] / e[4]),      # N: 32 -> 256
            "slope_tail": c["slope_tail"],
            "err_at_N256": float(e[-1]),
        }
    print(f"{'config':26s}{'8x steps':>12s}{'tail slope':>13s}")
    for k, v in table.items():
        print(f"{k:26s}{v['gain_8x_steps']:>11.2f}x{v['slope_tail']:>13.3f}")

    base = table["EM smooth"]["err_at_N256"]
    print("\n  schedule-side lever (no extra steps), error at N=256:")
    for k, v in table.items():
        print(f"    {k:26s}: {v['err_at_N256']:.3e}  ({v['err_at_N256']/base:5.2f}x of EM-smooth)")

    out = {
        "experiment": "E4 intervention: steps vs schedule vs score roughness",
        "config": {
            "data": "2D Gaussian mixture, mu=(-1.5,0),(1.5,0), sigma=I",
            "total_beta": TOTAL_BETA,
            "abar_T": float(abar_of("const", np.array([1.0]))[0]),
            "N_ref": N_REF, "n_paths": 200, "kappa": 2.0 / 3.0,
            "roughness": "amp * (beta(t)/beta_bar) * W_gamma(t) * x/|x|",
        },
        "cases": cases,
        "interventions": table,
    }
    payload = {"N": np.array(N_LIST)}
    for i, c in enumerate(out["cases"]):
        payload[f"err_{i}"] = np.array(c["err"])
        for N, arr in c["per_path"].items():
            payload[f"c{i}_N{N}"] = arr
    for c in out["cases"]:
        c.pop("per_path", None)
    with open(os.path.join(here, "e4_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(here, "e4_data.npz"), **payload)
    print("\n[E4] saved e4_result.json / e4_data.npz")


if __name__ == "__main__":
    main()
