"""
E6: 真实 score 网络上假设 4 的经验统计

目的
----
Proposition (充分条件) 说：时间模连续性给出 omega_N <= (L_t/2) N^{kappa-1}；
Proposition (反例) 说：若 score 在时间方向上有与采样网格共振的高频成分，
omega_N 不衰减。本实验在**真实训练出来的网络**上测量诊断量

    osc(N) = sup_{x,t} || (1/M) sum_{j<M} s(x, t+j/N) - s(x,t) ||,   M = floor(N^kappa)

用两个只有时间嵌入频率不同的网络做对照：
  - 低频网络 (K=1)：时间特征只有 sin(pi t), cos(pi t)，时间方向光滑
  - 高频网络 (K=8)：时间特征到 sin(128 pi t)，周期短至 1/64
其余一切（数据、宽度、深度、优化器、训练步数、种子）完全相同。

环境说明
--------
本机 PyTorch 的 shm.dll 依赖缺失，无法加载；因此用 numpy 手写一个三层 MLP
与反向传播。网络规模小（14x14 输入、192 宽、3 隐层），训练与评估全部在 CPU 上
完成，总耗时数分钟，结果可复现。
"""

import os
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "mnist.npy")

T_MAX = 1.0
T_MIN = 0.02


# ----------------------------------------------------------------- 数据
def load_mnist(n=12000, pool=2):
    X = np.load(DATA)[:n].astype(np.float32) / 255.0
    if pool > 1:
        n_, h, w = X.shape
        X = X.reshape(n_, h // pool, pool, w // pool, pool).mean(axis=(2, 4))
    X = X.reshape(X.shape[0], -1)
    X = X * 2.0 - 1.0                       # 归一到 [-1, 1]
    return X.astype(np.float32)


def abar(t):
    """VP-SDE, beta(t) = 6t + 0.1  =>  abar(t) = exp(-(3t^2 + 0.1t))"""
    return np.exp(-(3.0 * t ** 2 + 0.1 * t))


# ----------------------------------------------------------------- 网络
def silu(z):
    s = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    return z * s, s


def silu_grad(z, s):
    return s * (1.0 + z * (1.0 - s))


def time_feat(t, K):
    """t: (B,) -> (B, 2K)"""
    freqs = (2.0 ** np.arange(K)) * np.pi
    a = t[:, None] * freqs[None, :]
    return np.concatenate([np.sin(a), np.cos(a)], axis=1).astype(np.float32)


class ScoreNet:
    def __init__(self, d, K, hidden=192, seed=0):
        rng = np.random.default_rng(seed)
        self.K, self.d = K, 2 * K

        def sc(a, b):
            return (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)

        self.W1, self.b1, self.V1 = sc(d, hidden), np.zeros(hidden, np.float32), sc(2 * K, hidden) * 0.7
        self.W2, self.b2, self.V2 = sc(hidden, hidden), np.zeros(hidden, np.float32), sc(2 * K, hidden) * 0.7
        self.W3, self.b3, self.V3 = sc(hidden, hidden), np.zeros(hidden, np.float32), sc(2 * K, hidden) * 0.7
        self.W4, self.b4 = (sc(hidden, d) * np.sqrt(hidden / d)).astype(np.float32), np.zeros(d, np.float32)
        self.names = ["W1", "b1", "V1", "W2", "b2", "V2", "W3", "b3", "V3", "W4", "b4"]
        self.m = {n: np.zeros_like(getattr(self, n)) for n in self.names}
        self.v = {n: np.zeros_like(getattr(self, n)) for n in self.names}
        self.step = 0

    def forward(self, x, t):
        te = time_feat(t, self.K)
        z1 = x @ self.W1 + self.b1 + te @ self.V1
        a1, s1 = silu(z1)
        z2 = a1 @ self.W2 + self.b2 + te @ self.V2
        a2, s2 = silu(z2)
        z3 = a2 @ self.W3 + self.b3 + te @ self.V3
        a3, s3 = silu(z3)
        out = a3 @ self.W4 + self.b4
        return out, (x, te, z1, a1, s1, z2, a2, s2, z3, a3, s3)

    def backward(self, cache, dout):
        x, te, z1, a1, s1, z2, a2, s2, z3, a3, s3 = cache
        g = {}
        g["W4"] = a3.T @ dout
        g["b4"] = dout.sum(0)
        da3 = dout @ self.W4.T
        dz3 = da3 * silu_grad(z3, s3)
        g["W3"] = a2.T @ dz3
        g["b3"] = dz3.sum(0)
        g["V3"] = te.T @ dz3
        da2 = dz3 @ self.W3.T
        dz2 = da2 * silu_grad(z2, s2)
        g["W2"] = a1.T @ dz2
        g["b2"] = dz2.sum(0)
        g["V2"] = te.T @ dz2
        da1 = dz2 @ self.W2.T
        dz1 = da1 * silu_grad(z1, s1)
        g["W1"] = x.T @ dz1
        g["b1"] = dz1.sum(0)
        g["V1"] = te.T @ dz1
        return g

    def adam(self, grads, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.step += 1
        for n in self.names:
            self.m[n] = b1 * self.m[n] + (1 - b1) * grads[n]
            self.v[n] = b2 * self.v[n] + (1 - b2) * (grads[n] ** 2)
            mh = self.m[n] / (1 - b1 ** self.step)
            vh = self.v[n] / (1 - b2 ** self.step)
            setattr(self, n, getattr(self, n) - lr * mh / (np.sqrt(vh) + eps))

    def score(self, x, t):
        """s(x,t) = -eps_hat(x,t) / sqrt(1 - abar(t))"""
        eps_hat, _ = self.forward(x, t)
        den = np.sqrt(np.maximum(1.0 - abar(t), 1e-3)).astype(np.float32)
        return -eps_hat / den[:, None]


# ----------------------------------------------------------------- 训练
def train(net, X, epochs=40, batch=256, seed=0, lr=2e-3):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot, cnt = 0.0, 0
        for i in range(0, n - batch + 1, batch):
            idx = perm[i:i + batch]
            x0 = X[idx]
            t = rng.uniform(T_MIN, T_MAX, size=len(idx)).astype(np.float32)
            a = abar(t)[:, None].astype(np.float32)
            eps = rng.standard_normal(x0.shape).astype(np.float32)
            xt = np.sqrt(a) * x0 + np.sqrt(1.0 - a) * eps
            out, cache = net.forward(xt, t)
            diff = out - eps
            loss = float((diff ** 2).mean())
            tot += loss * len(idx)
            cnt += len(idx)
            g = net.backward(cache, 2.0 * diff / diff.size)
            net.adam(g, lr=lr)
        hist.append(tot / cnt)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    ep {ep:3d}  loss {hist[-1]:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    return hist


# ----------------------------------------------------------------- 诊断量
def measure_osc(net, X_eval, N_list, kappa=2.0 / 3.0, n_t=10,
                t_lo=0.10, t_hi=0.80, n_x=128, seed=1):
    """osc(N) = mean over (x,t) of || window-avg - s(x,t) || / || s(x,t) ||
    同时给出 95 分位，避免只看均值。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_eval.shape[0], size=min(n_x, X_eval.shape[0]), replace=False)
    xb = X_eval[idx]
    ts = np.linspace(t_lo, t_hi, n_t).astype(np.float32)
    out = {}
    for N in N_list:
        M = max(2, int(np.floor(N ** kappa)))
        rel, raw = [], []
        for t0 in ts:
            js = np.arange(M) / N
            if t0 + js[-1] > T_MAX:
                continue
            acc = np.zeros_like(xb, dtype=np.float32)
            for j in js:
                acc += net.score(xb, np.full(len(xb), t0 + j, dtype=np.float32))
            avg = acc / M
            s0 = net.score(xb, np.full(len(xb), t0, dtype=np.float32))
            num = np.linalg.norm((avg - s0).reshape(len(xb), -1), axis=1)
            den = np.linalg.norm(s0.reshape(len(xb), -1), axis=1) + 1e-8
            rel.extend((num / den).tolist())
            raw.extend(num.tolist())
        rel = np.array(rel)
        out[N] = {
            "M": M,
            "mean_rel": float(rel.mean()),
            "p95_rel": float(np.percentile(rel, 95)),
            "mean_abs": float(np.mean(raw)),
        }
        print(f"    N={N:4d}  M={M:4d}  rel osc = {out[N]['mean_rel']:.4e} "
              f"(p95 {out[N]['p95_rel']:.4e})", flush=True)
    return out


def measure_time_holder(net, X_eval, deltas, n_x=128, n_t=12,
                        t_lo=0.10, t_hi=0.80, seed=2):
    """测 score 的时间 Holder 指数：
         Delta_s(delta) = mean_{x,t} || s(x,t+delta) - s(x,t) || / mean||s||
       若 Delta_s ~ delta^{gamma_s}，则由窗口平均得
         osc(N) ~ N^{-gamma_s (1-kappa)}
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_eval.shape[0], size=min(n_x, X_eval.shape[0]), replace=False)
    xb = X_eval[idx]
    ts = np.linspace(t_lo, t_hi, n_t).astype(np.float32)
    scale = np.mean([np.linalg.norm(net.score(xb, np.full(len(xb), t0, dtype=np.float32)),
                                   axis=1).mean() for t0 in ts])
    out = {}
    for dl in deltas:
        vals = []
        for t0 in ts:
            if t0 + dl > T_MAX:
                continue
            s0 = net.score(xb, np.full(len(xb), t0, dtype=np.float32))
            s1 = net.score(xb, np.full(len(xb), t0 + dl, dtype=np.float32))
            vals.extend(np.linalg.norm(s1 - s0, axis=1).tolist())
        out[float(dl)] = float(np.mean(vals) / scale)
    return out, float(scale)


def main():
    print("[E6] loading MNIST (14x14, [-1,1])")
    X = load_mnist(30000, pool=2)
    print("   shape", X.shape, "range", X.min(), X.max())
    d = X.shape[1]

    N_LIST = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    results = {}

    for K in (1, 8):
        tag = f"K{K}"
        print(f"\n[E6] training network with time-embedding depth K={K} "
              f"(max temporal frequency = {2**(K-1)}*pi)")
        net = ScoreNet(d, K=K, hidden=192, seed=100 + K)
        hist = train(net, X, epochs=150, batch=256, seed=200 + K)
        print(f"[E6] K={K}: final DSM loss = {hist[-1]:.5f} (init {hist[0]:.4f})")
        print(f"[E6] measuring time-Holder exponent for K={K}")
        DELTAS = [1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4]
        hs, sc = measure_time_holder(net, X, DELTAS)
        for dl, v in hs.items():
            print(f"    delta={dl:.5f}  |Ds|/|s| = {v:.4e}")
        dd = np.array(sorted(hs)); yy = np.array([hs[x] for x in dd])
        gamma_s, _ = np.polyfit(np.log(dd), np.log(yy), 1)
        print(f"[E6] K={K}: fitted time-Holder exponent gamma_s = {gamma_s:.3f}")

        print(f"[E6] measuring osc(N) for K={K}")
        osc = measure_osc(net, X, N_LIST)
        results[tag] = {
            "K": K, "final_loss": float(hist[-1]),
            "loss_history": hist[-1:],
            "delta_scan": hs, "gamma_s": float(gamma_s), "score_scale": sc,
            "osc": {str(k): v for k, v in osc.items()},
        }

    # 拟合斜率
    for tag, r in results.items():
        ns = np.array(sorted(int(k) for k in r["osc"]), dtype=float)
        ys = np.array([r["osc"][str(int(n))]["mean_rel"] for n in ns])
        sl, _ = np.polyfit(np.log(ns), np.log(ys), 1)
        r["fitted_slope"] = float(sl)
        kappa = 2.0 / 3.0
        pred = -r["gamma_s"] * (1 - kappa)
        r["predicted_slope"] = float(pred)
        print(f"[E6] {tag}: osc slope = {sl:+.3f} ; "
              f"predicted from gamma_s={r['gamma_s']:.3f} is {pred:+.3f}")

    results["config"] = {
        "data": "MNIST 14x14, 30000 samples, normalized to [-1,1]",
        "sde": "VP, beta(t)=6t+0.1, T=1",
        "training": "150 epochs, batch 256, Adam lr=2e-3, eps-prediction DSM",
        "kappa": 2.0 / 3.0,
        "n_eval_images": 128,
    }
    with open(os.path.join(HERE, "e6_result.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n[E6] saved e6_result.json")


if __name__ == "__main__":
    main()
