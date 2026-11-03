"""차: 어떻게 움직이는지.

단순화한 자전거 모델. 모든 트랙에서 같은 상수를 쓴다.
핵심 규칙: 빠를수록 많이 못 꺾는다(횡가속 한계) → 코너 전에 감속해야 함.
"""
import numpy as np

MAX_SPEED = 260.0      # px/s
MAX_REVERSE = 60.0     # 후진 최대 속도
ACCEL = 320.0          # 가속력 (px/s^2)
BRAKE = 520.0          # throttle 음수일 때 감속력
DRAG = 0.6             # 공기저항/마찰 계수 (1/s)
MAX_YAW = 3.2          # 저속 최대 회전 속도 (rad/s)
LAT_GRIP = 420.0       # 횡가속 한계 (px/s^2). 회전속도 <= LAT_GRIP / 속도
CAR_LENGTH = 22.0
CAR_WIDTH = 12.0

# 전략 모드 전용 (일반 트랙에선 안 쓰임 → 기존 결과와 완전히 같음)
BOOST_ACCEL = 1.9      # 부스트 중 가속력 배수
BOOST_SPEED = 1.35     # 부스트 중 최고속도 배수
BOOST_GRIP = 0.75      # 부스트 중 그립 배수 (빠르지만 덜 꺾임)
OVERSPEED_DECAY = 260  # 부스트 끝난 뒤 최고속도 위로 남은 속도가 줄어드는 속도 (px/s^2)


class Car:
    def __init__(self, x, y, heading):
        self.x, self.y, self.heading = float(x), float(y), float(heading)
        self.speed = 0.0
        self.yaw_rate = 0.0

    def update(self, steer, throttle, dt, boost=False, grip=1.0):
        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, -1.0, 1.0))
        prev = self.speed

        # 가속 / 브레이크
        if throttle >= 0:
            self.speed += throttle * ACCEL * (BOOST_ACCEL if boost else 1.0) * dt
        elif self.speed > 0:
            self.speed = max(0.0, self.speed + throttle * BRAKE * dt)
        else:
            self.speed += throttle * ACCEL * 0.5 * dt   # 멈춘 상태에서 음수 = 후진
        self.speed -= self.speed * DRAG * dt
        cap = MAX_SPEED * (BOOST_SPEED if boost else 1.0)
        if self.speed > cap and prev > cap:             # 부스트가 끝나 과속 → 서서히 감속
            self.speed = max(cap, self.speed - OVERSPEED_DECAY * dt)
        else:
            self.speed = float(np.clip(self.speed, -MAX_REVERSE, cap))

        # 조향: 거의 멈춰 있으면 못 돌고, 빠르면 그립 한계에 걸림
        v = abs(self.speed)
        g = LAT_GRIP * grip * (BOOST_GRIP if boost else 1.0)
        yaw_limit = min(MAX_YAW * min(v / 40.0, 1.0), g / max(v, 1.0))
        self.yaw_rate = steer * yaw_limit * np.sign(self.speed or 1.0)
        self.heading += self.yaw_rate * dt
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi

        self.x += np.cos(self.heading) * self.speed * dt
        self.y += np.sin(self.heading) * self.speed * dt

    @property
    def pos(self):
        return np.array([self.x, self.y])
