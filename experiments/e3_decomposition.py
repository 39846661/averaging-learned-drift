"""
E3: 三项误差分解的数值验证（支撑 Theorem 3.5 与 Corollary 3.6）
—— 配对耦合版本

为什么必须配对
--------------
第一版用 sliced-W2 测终端分布，结果被估计偏差淹没（3000 样本的地板约 0.046，
真实信号只有 0.005）。这里改为：同一条反向布朗路径同时驱动"粗解"和"细解"，
误差定义在轨道级，不存在统计噪声，且与 Theorem 3.5 的 ||.||_{gamma} 口径一致。

设定
----
数据   : 二维高斯混合 p0 = 1/2 N((-1.5,0), I) + 1/2 N((1.5,0), I)
前向   : VP-SDE, beta(t) = 6t + 0.1, T = 1  =>  p_t(x) = sum_i w_i N(x; sqrt(abar_t) mu_i, I)
反向   : dx = [-1/2 beta(t) x - beta(t) s(x,t)] dt + sqrt(beta(t)) dWbar
score  : 闭式；s_theta = s + delta * phi(x)，phi(x)=x/sqrt(1+|x|^2)，sup|phi| <= 1
         因此 delta 恰是 Hypothesis 3 中的 delta_theta

子实验
------
E3a  固定 delta=0，扫步数 N：验证 (C) 项对 dt 的阶
E3b  固定 N，扫 delta：验证 (B) 项对 delta_theta 的线性
"""

import os
import json
import numpy as np

RNG = np.random.default_rng(20261101)

MU = np.array([[-1.5, 0.0], [1.5, 0.0]])
W = np.array([0.5, 0.5])
D = 2
T_MAX = 1.0
N_REF = 4096


def beta(t):
    return 6.0 * t + 0.1


def abar(t):
    return np.exp(-(3.0 * t ** 2 + 0.1 * t))


def score_true(x, t):
    a = abar(t)
    m = np.sqrt(a) * MU
    d2 = ((x[:, None, :] - m[None, :, :]) ** 2).sum(axis=2)
    lp = np.log(W)[None, :] - 0.5 * d2
    lp -= lp.max(axis=1, keepdims=True)
    wgt = np.exp(lp)
    wgt /= wgt.sum(axis=1, keepdims=True)
    return -x + np.sqrt(a) * (wgt @ MU)


def phi(x):
    return x / np.sqrt(1.0 + (x ** 2).sum(axis=1, keepdims=True))


def reverse_sde(xT, n_steps, xi, delta=0.0):
    """反向 SDE 的 Euler-Maruyama，xi 为 (M, n_steps) 标准正态"""
    x = xT.copy()
    ts = np.linspace(T_MAX, 0.0, n_steps + 1)
    for i in range(n_steps):
        t = ts[i]
        h = abs(ts[i + 1] - ts[i])
        b0 = 0.5 * beta(t) * x
        s = score_true(x, t) + (delta * phi(x) if delta > 0 else 0.0)
        x = x - (b0 + beta(t) * s) * h + np.sqrt(beta(t) * h) * xi[:, i, None]
    return x


def main():
    M = 2000
    print("[E3] paired coupling: same Brownian path for coarse and reference")

    # 从 p_T 采样初值
    aT = abar(T_MAX)
    idx = RNG.choice(2, size=M, p=W)
    xT = np.sqrt(aT) * MU[idx] + RNG.standard_normal((M, D))

    xi_ref = RNG.standard_normal((M, N_REF))
    print(f"[E3] building reference (N_ref={N_REF}) ...")
    xref = reverse_sde(xT, N_REF, xi_ref, delta=0.0)

    # ---------------- E3a: (C) 项 ----------------
    print("\n[E3a] delta_theta = 0, scan N")
    N_LIST = [16, 32, 64, 128, 256, 512, 1024]
    err_a = []
    per_path, per_delta = {}, {}
    for N in N_LIST:
        block = N_REF // N
        # 粗网格用同样的布朗路径：块和 = sqrt(block) * 平均
        xi_c = xi_ref[:, : block * N].reshape(M, N, block).sum(axis=2) / np.sqrt(block)
        xe = reverse_sde(xT, N, xi_c, delta=0.0)
        ei = np.linalg.norm(xe - xref, axis=1)
        per_path[N] = ei
        e = ei.mean()
        err_a.append(float(e))
        print(f"  N={N:5d}  dt={T_MAX/N:.5f}  E||X^N - X_ref|| = {e:.5e}")

    ns = np.array(N_LIST, dtype=float)
    sl_a, _ = np.polyfit(np.log(ns), np.log(err_a), 1)
    print(f"[E3a] fitted slope = {sl_a:+.4f}  (order in dt = {-sl_a:+.4f})")

    # ---------------- E3b: (B) 项 ----------------
    print("\n[E3b] N = 512, scan delta_theta")
    N0 = 512
    block = N_REF // N0
    xi_c = xi_ref[:, : block * N0].reshape(M, N0, block).sum(axis=2) / np.sqrt(block)
    D_LIST = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
    err_b = []
    for dl in D_LIST:
        xe = reverse_sde(xT, N0, xi_c, delta=dl)
        ei = np.linalg.norm(xe - xref, axis=1)
        per_delta[dl] = ei
        e = ei.mean()
        err_b.append(float(e))
        print(f"  delta_theta={dl:5.3f}   E||X - X_ref|| = {e:.5e}")

    ds = np.array(D_LIST[1:], dtype=float)
    sl_b, ic_b = np.polyfit(ds, np.array(err_b[1:]), 1)
    print(f"[E3b] linear fit on delta>0: slope = {sl_b:+.4f}, intercept = {ic_b:+.5e}")

    out = {
        "experiment": "E3 three-term decomposition (paired coupling)",
        "data": "2D Gaussian mixture mu=(-1.5,0),(1.5,0), sigma=I",
        "sde": "VP: beta(t)=6t+0.1, T=1, reverse EM",
        "N_ref": N_REF, "n_paths": M,
        "E3a_dt_scan": {"N": N_LIST, "err": err_a, "fitted_slope": float(sl_a)},
        "E3b_delta_scan": {"delta": D_LIST, "err": err_b,
                           "linear_slope": float(sl_b), "linear_intercept": float(ic_b)},
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "e3_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(here, "e3_data.npz"),
             N=np.array(N_LIST), err_dt=np.array(err_a),
             delta=np.array(D_LIST), err_delta=np.array(err_b),
             **{f"paths_{n}": per_path[n] for n in N_LIST},
             **{f"dpaths_{k}": per_delta[k] for k in D_LIST})
    print("\n[E3] saved e3_result.json / e3_data.npz")


if __name__ == "__main__":
    main()
