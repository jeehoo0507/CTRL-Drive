"""제출용 에이전트 템플릿.

규칙은 딱 두 개:
  1. __init__() : 가중치 로드 (한 번)
  2. act(obs)   : 관측 12개 → [steer, throttle] (매 프레임)

이 예시는 train.py가 만든 weights.npz를 numpy만으로 실행한다.
(평가 서버에 PyTorch가 없어도 돌아감)
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


class Agent:
    def __init__(self):
        w = np.load(os.path.join(HERE, "weights.npz"))
        self.layers = [(w[f"w{i}"], w[f"b{i}"]) for i in range(len(w.files) // 2)]

    def act(self, obs):
        x = np.asarray(obs, dtype=np.float32)[: self.layers[0][0].shape[1]]   # 가중치가 받는 만큼만
        for i, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return np.clip(x, -1.0, 1.0)      # [steer, throttle]
