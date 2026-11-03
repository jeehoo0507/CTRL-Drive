"""베이스라인 학습: Stable-Baselines3 PPO.

    python train.py                      # 기본 30만 스텝, 연습 트랙 전부
    python train.py --timesteps 1000000  # 더 길게

결과: weights.npz (agent.py가 읽는 파일)
여기부터는 자유: 보상 바꾸기(reward_fn), 하이퍼파라미터, 다른 알고리즘 전부 OK.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import racegame                                       # noqa: E402
from racegame.gym_wrapper import GymRaceEnv          # noqa: E402
from stable_baselines3 import PPO                    # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env   # noqa: E402


def my_reward(obs, action, reward, info):
    """보상 설계 예시. 기본 보상(전진 거리, 충돌 -10)에 살짝 덧붙임."""
    return reward - 0.01                     # 시간 페널티 → 빨리 달리도록


def export_npz(model, path):
    """SB3 정책망(행동 부분만) → numpy 가중치. 평가 땐 PyTorch 불필요."""
    sd = model.policy.state_dict()
    keys = sorted(k for k in sd if k.startswith("mlp_extractor.policy_net") and "weight" in k)
    arrays, i = {}, 0
    for k in keys:
        arrays[f"w{i}"] = sd[k].cpu().numpy()
        arrays[f"b{i}"] = sd[k.replace("weight", "bias")].cpu().numpy()
        i += 1
    arrays[f"w{i}"] = sd["action_net.weight"].cpu().numpy()
    arrays[f"b{i}"] = sd["action_net.bias"].cpu().numpy()
    np.savez(path, **arrays)


class Progress(BaseCallback):
    """중간 저장 + 진행 로그 (코랩이 끊겨도 가중치는 남게)."""

    def __init__(self, out, every=50_000):
        super().__init__()
        self.out, self.every, self.next, self.t0 = out, every, every, time.time()

    def _on_step(self):
        if self.num_timesteps >= self.next:
            self.next += self.every
            export_npz(self.model, self.out)
            ep = [e["r"] for e in self.model.ep_info_buffer]
            print(f"[{self.num_timesteps:>8,} steps | {time.time() - self.t0:5.0f}s] "
                  f"평균 에피소드 보상 {np.mean(ep) if ep else 0:6.1f}  → {os.path.basename(self.out)} 저장",
                  flush=True)
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--tracks", nargs="+", default=racegame.list_tracks())
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--out", default=os.path.join(HERE, "weights.npz"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    print(f"학습 트랙: {a.tracks} | 스텝: {a.timesteps:,}")
    venv = make_vec_env(GymRaceEnv, n_envs=a.envs, seed=a.seed,
                        env_kwargs=dict(tracks=a.tracks, reward_fn=my_reward))
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=512, n_epochs=10,
                learning_rate=3e-4, gamma=0.995, gae_lambda=0.95, ent_coef=0.0,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
                seed=a.seed, verbose=0, device="cpu")
    model.learn(total_timesteps=a.timesteps, callback=Progress(a.out))
    export_npz(model, a.out)
    print(f"완료 → {a.out}")

    agent = racegame.load_agent(HERE)
    print(racegame.format_table(racegame.evaluate(agent, a.tracks)))


if __name__ == "__main__":
    main()
