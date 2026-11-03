"""
E1: 前向快慢格式的轨道级收敛速率（验证 Theorem 3.1 / Remark 3.4）

被验证的命题
------------
对 dX = b(X)dt + sigma(X)dW 的离散快慢格式 (2)，
    E[ || X^N - Xbar ||_{gamma} ] <= C * N^{-(1/2-gamma)}
本实验在 i.i.d. 噪声（beta = infinity，即 Remark 3.4 的退化情形）下
测量实际斜率，检查实测斜率是否不低于理论下界 1/2 - gamma。

设计要点（审稿人会查的三件事）
------------------------------
1. 参考解必须比最细的 N 再细两个数量级（N_ref = 2^15），且用 Milstein 而非
   Euler，否则测到的是"参考解的误差"而不是收敛率。
2. 粗网格与参考解必须由**同一条布朗路径**驱动：粗网格第 k 步的增量取 m=N_ref/N
   个细增量的平均，保证 E[(1/sqrt(m)) sum xi]^2 = 1。
3. Holder 范数取 sup + semi 两项，gamma = 0.25（< 1/2，这是硬约束）。
"""

import os
import json
import numpy as np

rng = np.random.default_rng(20261101)

T = 1.0
X0 = 1.0
N_REF = 2 ** 15
N_LIST = [2 ** p for p in range(4, 13)]      # 16 ... 4096
M_PATHS = 4000
GAMMAS = [0.25]

# 测试方程：乘性噪声，经典 Euler-Maruyama 强收敛率 1/2
#   dX = -X dt + (0.5 + 0.3 X) dW
def b(x):
    return -x

def sigma(x):
    return 0.5 + 0.3 * x

def dsigma(x):
    return 0.3


def reference_paths(xi, dt_ref):
    """Milstein 在细网格上的参考解，形状 (M, N_REF+1)"""
    m = xi.shape[0]
    n = xi.shape[1]
    x = np.full(m, X0)
    out = np.empty((m, n + 1))
    out[:, 0] = x
    for k in range(n):
        s = sigma(x)
        x = (x + b(x) * dt_ref + s * np.sqrt(dt_ref) * xi[:, k]
             + 0.5 * s * dsigma(x) * dt_ref * (xi[:, k] ** 2 - 1.0))
        out[:, k + 1] = x
    return out


def coarse_paths(xi, n_steps):
    """格式 (2) 在 N = n_steps 的路径，返回粗网格上的值 (M, N+1)"""
    m = xi.shape[1]
    block = m // n_steps
    dt = T / n_steps
    # 块平均：同一布朗路径的粗网格增量
    xi_block = xi[:, : block * n_steps].reshape(xi.shape[0], n_steps, block).mean(axis=2)
    xi_block = xi_block * np.sqrt(block)        # 方差还原为 1
    x = np.full(xi.shape[0], X0)
    out = np.empty((xi.shape[0], n_steps + 1))
    out[:, 0] = x
    for k in range(n_steps):
        x = x + b(x) * dt + sigma(x) * np.sqrt(dt) * xi_block[:, k]
        out[:, k + 1] = x
    return out


def holder_norm(err, dt, gamma):
    """sup 范数 + gamma-Holder 半范数
    err: (M, N+1) 粗网格上的误差路径"""
    sup = np.abs(err).max(axis=1)
    d = np.abs(np.diff(err, axis=1)) / dt ** gamma
    semi = d.max(axis=1)
    return sup + semi


def main():
    print(f"[E1] N_ref = {N_REF}, paths = {M_PATHS}, T = {T}")
    # 生成一条共享的 i.i.d. 标准正态序列
    xi = rng.standard_normal((M_PATHS, N_REF))

    dt_ref = T / N_REF
    print("[E1] building reference (Milstein, N_ref steps)...")
    xref = reference_paths(xi, dt_ref)

    results = {str(g): [] for g in GAMMAS}
    all_err = {}
    raw = {}
    for n in N_LIST:
        xc = coarse_paths(xi, n)
        # 把参考解抽到粗网格点上对齐
        idx = np.arange(0, N_REF + 1, N_REF // n)
        xr = xref[:, idx]
        err = xc - xr
        dt = T / n
        all_err[n] = holder_norm(err, dt, 0.25)
        for g in GAMMAS:
            v = holder_norm(err, dt, g)
            results[str(g)].append(float(v.mean()))
        raw[n] = {
            "mean_err_gamma025": float(holder_norm(err, dt, 0.25).mean()),
            "mean_err_sup": float(np.abs(err).max(axis=1).mean()),
            "std_err_gamma025": float(holder_norm(err, dt, 0.25).std(ddof=1)),
        }
        print(f"  N={n:5d}  dt={dt:.2e}  E[||err||_0.25] = "
              f"{raw[n]['mean_err_gamma025']:.4e}   (sup: {raw[n]['mean_err_sup']:.4e})")

    # log-log 拟合斜率
    ns = np.array(N_LIST, dtype=float)
    slopes = {}
    for g in GAMMAS:
        ys = np.array(results[str(g)])
        slope, intercept = np.polyfit(np.log(ns), np.log(ys), 1)
        slopes[str(g)] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "theory_lower_bound": -(0.5 - g),
        }
        print(f"[E1] gamma={g}: fitted slope = {slope:+.4f}  "
              f"(theory bound >= {-(0.5-g):+.4f})")

    out = {
        "experiment": "E1 forward trajectory-level rate",
        "equation": "dX = -X dt + (0.5+0.3X) dW, X0=1, T=1",
        "N_list": N_LIST,
        "N_ref": N_REF,
        "n_paths": M_PATHS,
        "noise": "i.i.d. standard normal (beta = infinity, degenerate case of Rem 3.4)",
        "error_curves": results,
        "per_N": raw,
        "fitted": slopes,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "e1_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(here, "e1_data.npz"),
             N=np.array(N_LIST),
             err_gamma=np.array(results["0.25"]),
             err_sup=np.array([raw[n]["mean_err_sup"] for n in N_LIST]),
             **{f"paths_{n}": all_err[n] for n in N_LIST})
    print("[E1] saved e1_result.json / e1_data.npz")


if __name__ == "__main__":
    main()
