"""
E2: 嵌入命题的余项阶（验证 Proposition 2.2）

被验证的命题
------------
DDPM 前向链 X_k = sqrt(1-beta_k) X_{k-1} + sqrt(beta_k) eps_k
与离散快慢格式 Y_k = (1 - beta_k/2) Y_{k-1} + sqrt(beta_k) eps_k
相差一个 O(N^{-2}) 阶的加性确定性余项。

这个实验不需要训练、不需要数据、没有统计噪声：
同一 eps 序列同时驱动两条递推，差距完全由 sqrt(1-b) 的 Taylor 展开决定。

关键问题（正文本需要修正的地方）
------------------------------
per-step 余项是 O(N^{-2})，但它在 N 步上是**累积**的。
若累积后仍是 O(N^{-2})，斜率应为 -2；若累积成 O(N^{-1})，斜率应为 -1。
实验直接给出答案，正文按实测阶数改口径。
"""

import os
import json
import numpy as np


def beta_schedule(t, beta0=0.1, beta1=10.0):
    """连续调度 beta(t) = beta0 + (beta1-beta0) t，t in [0,1]"""
    return beta0 + (beta1 - beta0) * t


def run(N, M, rng, T=1.0, x0=1.0):
    """返回 sup_k |X_k - Y_k|，形状 (M,)"""
    k = np.arange(1, N + 1, dtype=float)
    bk = beta_schedule(k / N) / N              # beta_k = beta(k/N)/N
    eps = rng.standard_normal((M, N))

    x = np.full(M, x0)      # DDPM 前向链
    y = np.full(M, x0)      # 快慢格式
    diff = np.empty((M, N + 1))
    diff[:, 0] = 0.0
    for i in range(N):
        x = np.sqrt(1.0 - bk[i]) * x + np.sqrt(bk[i]) * eps[:, i]
        y = (1.0 - 0.5 * bk[i]) * y + np.sqrt(bk[i]) * eps[:, i]
        diff[:, i + 1] = x - y
    return np.abs(diff).max(axis=1), np.abs(diff[:, -1])


def main():
    rng = np.random.default_rng(2026)
    N_LIST = [2 ** p for p in range(5, 16)]     # 32 ... 32768
    M = 2000
    err, term = [], []
    print("[E2] DDPM forward chain vs. fast-slow scheme, shared eps")
    for N in N_LIST:
        e, t = run(N, M, rng)
        err.append(float(e.mean()))
        term.append(float(t.mean()))
        print(f"  N={N:6d}  sup_k|X-Y| = {e.mean():.4e}   |X_T-Y_T| = {t.mean():.4e}")

    ns = np.array(N_LIST, dtype=float)
    s_sup, _ = np.polyfit(np.log(ns), np.log(err), 1)
    s_term, _ = np.polyfit(np.log(ns), np.log(term), 1)
    print(f"[E2] fitted slope (sup over k) : {s_sup:+.4f}")
    print(f"[E2] fitted slope (terminal)   : {s_term:+.4f}")

    out = {
        "experiment": "E2 embedding remainder order",
        "schedule": "beta(t)=0.1+9.9t, beta_k = beta(k/N)/N, T=1",
        "N_list": N_LIST,
        "n_paths": M,
        "sup_error": err,
        "terminal_error": term,
        "fitted_slope_sup": float(s_sup),
        "fitted_slope_terminal": float(s_term),
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "e2_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(here, "e2_data.npz"),
             N=np.array(N_LIST), sup=np.array(err), term=np.array(term))
    print("[E2] saved e2_result.json / e2_data.npz")


if __name__ == "__main__":
    main()
