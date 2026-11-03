"""센서: 차가 길을 어떻게 보는지.

관측(obs) 벡터 12개 (모두 대략 -1~1 범위로 정규화):
  [0:9]  레이캐스트 9개: 차 기준 -90°~+90°, 벽까지 거리 / RAY_MAX
  [9]    속도 / MAX_SPEED
  [10]   100px 앞 중심선 지점의 방향 - 차 방향 (/pi)
  [11]   250px 앞 중심선 지점의 방향 - 차 방향 (/pi)
"""
import numpy as np
from .car import MAX_SPEED

N_RAYS = 9
RAY_FOV = np.radians(180)
RAY_MAX = 300.0
RAY_ANGLES = np.linspace(-RAY_FOV / 2, RAY_FOV / 2, N_RAYS)
LOOKAHEAD = (100.0, 250.0)
OBS_DIM = N_RAYS + 1 + len(LOOKAHEAD)


def raycast(origin, heading, wall_a, wall_b, max_dist=RAY_MAX):
    """모든 레이 x 모든 벽 선분 교차를 한 번에 계산. 반환: 거리 배열 (N_RAYS,)"""
    ang = heading + RAY_ANGLES
    d = np.stack([np.cos(ang), np.sin(ang)], axis=1)          # (R, 2)
    e = wall_b - wall_a                                       # (W, 2)
    ao = wall_a - origin                                      # (W, 2)
    # origin + t*d = a + u*e  를 t, u에 대해 풂
    denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]   # (R, W)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ao[None, :, 0] * e[None, :, 1] - ao[None, :, 1] * e[None, :, 0]) / denom
        u = (ao[None, :, 0] * d[:, None, 1] - ao[None, :, 1] * d[:, None, 0]) / denom
    hit = (np.abs(denom) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
    t = np.where(hit, t, max_dist)
    return np.minimum(t.min(axis=1), max_dist)


def raycast_circles(origin, heading, circles, max_dist=RAY_MAX):
    """레이 x 원형 장애물 교차. circles: (M, 3) [x, y, r] → (N_RAYS,)"""
    if circles is None or len(circles) == 0:
        return np.full(N_RAYS, max_dist)
    ang = heading + RAY_ANGLES
    d = np.stack([np.cos(ang), np.sin(ang)], axis=1)          # (R, 2)
    oc = circles[:, :2] - origin                              # (M, 2)
    b = d @ oc.T                                              # (R, M) 레이 방향 투영
    c = (oc ** 2).sum(1) - circles[:, 2] ** 2                 # (M,)
    disc = b ** 2 - c[None]
    with np.errstate(invalid="ignore"):
        t = b - np.sqrt(disc)
    t = np.where((disc >= 0) & (t > 0), t, max_dist)
    t = np.where(c[None] < 0, 0.0, t)                         # 이미 안에 있으면 0
    return np.minimum(t.min(axis=1), max_dist)


def angle_to(car, target):
    dx, dy = target[0] - car.x, target[1] - car.y
    a = np.arctan2(dy, dx) - car.heading
    return (a + np.pi) % (2 * np.pi) - np.pi


def observe(car, track, s_on_track, obstacles=None, segments=None):
    """obstacles: 원형 장애물 (M,3) / segments: 박스 모서리 선분 (a, b) — 전부 레이에 잡힘."""
    rays = raycast(car.pos, car.heading, track.wall_a, track.wall_b)
    if segments is not None:
        rays = np.minimum(rays, raycast(car.pos, car.heading, segments[0], segments[1]))
    if obstacles is not None and len(obstacles):
        rays = np.minimum(rays, raycast_circles(car.pos, car.heading, obstacles))  # 장애물도 보임
    rays = rays / RAY_MAX
    look = [angle_to(car, track.point_at(s_on_track + d)) / np.pi for d in LOOKAHEAD]
    return np.concatenate([rays, [car.speed / MAX_SPEED], look]).astype(np.float32)
