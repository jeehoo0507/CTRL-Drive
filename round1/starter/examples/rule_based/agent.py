"""신경망 없이 if문만으로 만든 에이전트 예시.
'인터페이스만 지키면 안은 뭐든 된다'는 걸 보여주는 용도.
"""
import numpy as np


class Agent:
    def __init__(self):
        pass

    def act(self, obs):
        rays = obs[:9]           # 왼쪽(-90°) ... 정면(0°) ... 오른쪽(+90°)
        speed = obs[9]
        near, far = obs[10], obs[11]

        # 조향: 가까운 목표 방향으로 + 양쪽 벽 거리 차이로 보정
        steer = 2.5 * near + 1.5 * (rays[6] - rays[2])

        # 속도: 앞이 트여 있고 멀리 있는 커브가 완만할수록 빠르게
        target = 0.35 + 0.6 * rays[4] - 0.5 * abs(far)
        throttle = 1.0 if speed < target else -0.6
        return np.array([np.clip(steer, -1, 1), throttle])
