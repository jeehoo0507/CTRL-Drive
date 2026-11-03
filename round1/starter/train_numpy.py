"""PyTorch 없이 numpy만으로 짠 PPO. (train.py와 같은 weights.npz를 만든다)

    python train_numpy.py --timesteps 600000

구조: 정책망 12→64→64→2 (tanh) + 별도 가치망 12→64→64→1, Adam, GAE.
역전파를 손으로 짠 교육용 구현이라 '안에서 무슨 일이 일어나는지' 다 보인다.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import racegame  # noqa: E402


# ---------------------------------------------------------------- MLP
class MLP:
    def __init__(self, sizes, rng, out_gain):
        self.params = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            gain = out_gain if i == len(sizes) - 2 else np.sqrt(2)
            u, _, vt = np.linalg.svd(rng.standard_normal((n_out, n_in)), full_matrices=False)
            W = (u if n_out >= n_in else vt) * gain          # 직교 초기화
            self.params += [W, np.zeros(n_out)]

    def forward(self, x):
        self.cache = [x]
        n = len(self.params) // 2
        for i in range(n):
            W, b = self.params[2 * i], self.params[2 * i + 1]
            x = x @ W.T + b
            if i < n - 1:
                x = np.tanh(x)
            self.cache.append(x)
        return x

    def backward(self, dout):
        grads = [None] * len(self.params)
        n = len(self.params) // 2
        for i in reversed(range(n)):
            h_in = self.cache[i]
            grads[2 * i] = dout.T @ h_in
            grads[2 * i + 1] = dout.sum(0)
            if i > 0:
                dout = (dout @ self.params[2 * i]) * (1 - h_in ** 2)
        return grads


class Adam:
    def __init__(self, params, lr):
        self.p, self.lr, self.t = params, lr, 0
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]

    def step(self, grads, max_norm=0.5):
        norm = np.sqrt(sum((g ** 2).sum() for g in grads))
        scale = min(1.0, max_norm / (norm + 1e-8))
        self.t += 1
        for i, g in enumerate(grads):
            g = g * scale
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            self.p[i] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


# ---------------------------------------------------------------- 환경 묶음
class VecEnv:
    def __init__(self, tracks, n, rng):
        self.rng = rng
        self.envs = [racegame.make(tracks[i % len(tracks)]) for i in range(n)]
        self.pool = {t: [racegame.make(t) for _ in range(n)] for t in tracks}
        self.tracks = tracks
        self.obs = np.stack([e.reset(seed=int(rng.integers(1 << 30))) for e in self.envs])

    def step(self, actions):
        rew = np.zeros(len(self.envs))
        done = np.zeros(len(self.envs), bool)
        trunc_obs = {}
        finished = []
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            o, r, d, info = e.step(np.clip(a, -1, 1))
            rew[i] = r - 0.01                                 # 시간 페널티
            if d:
                finished.append(info)
                if not (info["crashed"] or info["finished"]):  # 시간초과/정체 = 잘린 것
                    trunc_obs[i] = o
                t = self.tracks[self.rng.integers(len(self.tracks))]
                e = self.envs[i] = self.pool[t][i]
                o = e.reset(seed=int(self.rng.integers(1 << 30)))
                done[i] = True
            self.obs[i] = o
        return self.obs.copy(), rew, done, trunc_obs, finished


class NpzAgent:
    """방금 저장한 가중치 파일로 바로 평가하기 위한 작은 에이전트 (agent.py 와 같은 계산)."""

    def __init__(self, path):
        w = np.load(path)
        self.layers = [(w[f"w{i}"], w[f"b{i}"]) for i in range(len(w.files) // 2)]

    def act(self, obs):
        x = np.asarray(obs, np.float32)[: self.layers[0][0].shape[1]]
        for i, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return np.clip(x, -1, 1)


# ---------------------------------------------------------------- PPO
def export(pi, path):
    np.savez(path, **{f"{'w' if j % 2 == 0 else 'b'}{j // 2}": p.astype(np.float32)
                      for j, p in enumerate(pi.params)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=600_000)
    ap.add_argument("--tracks", nargs="+", default=racegame.list_tracks())
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(HERE, "weights.npz"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    gamma, lam, clip, epochs, mb, lr, vf_scale = 0.99, 0.95, 0.2, 8, 1024, 3e-4, 0.1
    rng = np.random.default_rng(a.seed)
    D = racegame.OBS_DIM
    pi = MLP([D, 64, 64, 2], rng, out_gain=0.01)
    vf = MLP([D, 64, 64, 1], rng, out_gain=1.0)
    log_std = np.full(2, -0.5)
    opt_pi = Adam(pi.params + [log_std], lr)
    opt_vf = Adam(vf.params, lr)

    venv = VecEnv(a.tracks, a.envs, rng)
    N, T = a.envs, a.steps
    total, t0, it, recent = 0, time.time(), 0, []
    print(f"학습 트랙: {a.tracks} | 스텝: {a.timesteps:,} | numpy PPO", flush=True)

    while total < a.timesteps:
        it += 1
        buf_o = np.zeros((T, N, D)); buf_a = np.zeros((T, N, 2)); buf_lp = np.zeros((T, N))
        buf_r = np.zeros((T, N)); buf_d = np.zeros((T, N)); buf_v = np.zeros((T, N))
        obs = venv.obs.copy()
        for t in range(T):
            mu = pi.forward(obs)
            std = np.exp(log_std)
            act = mu + std * rng.standard_normal(mu.shape)
            lp = (-0.5 * ((act - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(1)
            v = vf.forward(obs)[:, 0] / vf_scale
            nobs, r, d, trunc, fin = venv.step(act)
            for i, o in trunc.items():                    # 잘린 에피소드는 가치로 부트스트랩
                r[i] += gamma * vf.forward(o[None])[0, 0] / vf_scale
            buf_o[t], buf_a[t], buf_lp[t], buf_r[t], buf_d[t], buf_v[t] = obs, act, lp, r, d, v
            recent += fin
            obs = nobs
        total += N * T

        # GAE
        last_v = vf.forward(obs)[:, 0] / vf_scale
        adv = np.zeros((T, N)); gae = 0
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else buf_v[t + 1]
            nonterm = 1 - buf_d[t]
            delta = buf_r[t] + gamma * nv * nonterm - buf_v[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + buf_v

        O, A, LP = buf_o.reshape(-1, D), buf_a.reshape(-1, 2), buf_lp.reshape(-1)
        ADV, RET = adv.reshape(-1), ret.reshape(-1) * vf_scale
        B = len(O)
        for _ in range(epochs):
            idx = rng.permutation(B)
            for s in range(0, B, mb):
                j = idx[s:s + mb]
                o, act, lp_old, ad, rt = O[j], A[j], LP[j], ADV[j], RET[j]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                # 정책
                mu = pi.forward(o)
                std = np.exp(log_std)
                z = (act - mu) / std
                lp = (-0.5 * z ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(1)
                ratio = np.exp(lp - lp_old)
                active = ((ad > 0) & (ratio < 1 + clip)) | ((ad < 0) & (ratio > 1 - clip))
                g_lp = -(ratio * ad * active) / len(j)           # d loss / d logp
                g_mu = g_lp[:, None] * z / std
                g_ls = (g_lp[:, None] * (z ** 2 - 1)).sum(0)
                opt_pi.step(pi.backward(g_mu) + [g_ls])
                # 가치
                v = vf.forward(o)[:, 0]
                g_v = (2 * (v - rt) / len(j))[:, None]
                opt_vf.step(vf.backward(g_v))
        np.clip(log_std, -2.5, 0.5, out=log_std)

        if it % 5 == 0 or total >= a.timesteps:
            export(pi, a.out)
            if recent:
                fin = np.mean([x["finished"] for x in recent]) * 100
                prog = np.mean([x["progress"] for x in recent]) * 100
                crash = np.mean([x["crashed"] for x in recent]) * 100
                print(f"[{total:>8,} steps | {time.time() - t0:5.0f}s] 진행률 {prog:5.1f}%  "
                      f"완주 {fin:5.1f}%  충돌 {crash:5.1f}%  std {np.exp(log_std).round(2)}",
                      flush=True)
            recent = []

    export(pi, a.out)
    print(f"완료 → {a.out}", flush=True)
    print(racegame.format_table(racegame.evaluate(NpzAgent(a.out), a.tracks)), flush=True)


if __name__ == "__main__":
    main()
