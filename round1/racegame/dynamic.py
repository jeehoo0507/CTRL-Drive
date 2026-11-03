"""움직이는 장애물 물리.

Boxes    : 밀 수 있는 박스(회전하는 정사각형). 차와 충격량(impulse) 기반 충돌 →
           가벼운 박스는 밀려나고(차는 감속), 무거운 박스는 거의 안 움직임(벽처럼).
           박스끼리, 박스-벽, 박스-콘 충돌도 처리. 마찰로 서서히 멈춤.
Shuttles : 트랙을 가로질러 일정 주기로 왕복하는 원형 장애물. 타이밍 맞춰 지나가야 함.

둘 다 박아도 탈락은 없고 막히기만 한다.
"""
import numpy as np

CAR_MASS = 1.0
BOX_RESTITUTION = 0.15      # 박스-차 반발 계수
BOX_DAMP = 2.5              # 박스 속도 감쇠 (1/s)
BOX_DECEL = 90.0            # 박스 바닥 마찰 감속 (px/s^2)
ANG_DAMP = 3.0              # 회전 감쇠 (1/s)
WALL_RESTITUTION = 0.2
CORNERS = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], float)


class Boxes:
    def __init__(self, items):
        a = np.array(items, float).reshape(-1, 5)
        self.pos = a[:, :2].copy()
        self.size = a[:, 2].copy()
        self.mass = a[:, 3].copy()
        self.ang = a[:, 4].copy()
        self.vel = np.zeros_like(self.pos)
        self.angv = np.zeros(len(a))
        self.inertia = self.mass * self.size ** 2 / 6.0
        self.idx = None

    def __len__(self):
        return len(self.pos)

    # ---------------------------------------------------------------- 형상
    def axes(self):
        c, s = np.cos(self.ang), np.sin(self.ang)
        return np.stack([c, s], 1), np.stack([-s, c], 1)          # u, v (B, 2)

    def corners(self):
        u, v = self.axes()
        hs = (self.size / 2)[:, None, None]
        return (self.pos[:, None] + hs * (CORNERS[None, :, :1] * u[:, None]
                                          + CORNERS[None, :, 1:] * v[:, None]))   # (B, 4, 2)

    def segments(self):
        c = self.corners()
        return c.reshape(-1, 2), np.roll(c, -1, axis=1).reshape(-1, 2)

    # ---------------------------------------------------------------- 차와 충돌
    def collide_car(self, car, car_r):
        if len(self) == 0:
            return False
        u_all, v_all = self.axes()
        hit = False
        for k in range(len(self)):
            u, v, hs = u_all[k], v_all[k], self.size[k] / 2
            d = car.pos - self.pos[k]
            lx, ly = d @ u, d @ v
            if abs(lx) < hs and abs(ly) < hs:                     # 차 중심이 박스 안
                if hs - abs(lx) < hs - abs(ly):
                    nl, cx, cy = np.array([np.sign(lx) or 1.0, 0.0]), np.sign(lx) * hs, ly
                    pen = hs - abs(lx) + car_r
                else:
                    nl, cx, cy = np.array([0.0, np.sign(ly) or 1.0]), lx, np.sign(ly) * hs
                    pen = hs - abs(ly) + car_r
            else:
                cx, cy = np.clip(lx, -hs, hs), np.clip(ly, -hs, hs)
                diff = np.array([lx - cx, ly - cy])
                dist = np.hypot(*diff)
                if dist >= car_r:
                    continue
                nl, pen = diff / max(dist, 1e-6), car_r - dist
            hit = True
            n = nl[0] * u + nl[1] * v                              # 박스 → 차 방향 법선
            contact = self.pos[k] + cx * u + cy * v
            mb, I = self.mass[k], self.inertia[k]
            # 위치 분리 (질량비대로)
            tot = CAR_MASS + mb
            car.x += n[0] * pen * mb / tot
            car.y += n[1] * pen * mb / tot
            self.pos[k] -= n * pen * CAR_MASS / tot
            # 속도: 충격량 교환 (회전 포함)
            hdir = np.array([np.cos(car.heading), np.sin(car.heading)])
            vc = car.speed * hdir
            r = contact - self.pos[k]
            vb = self.vel[k] + self.angv[k] * np.array([-r[1], r[0]])
            vrel = (vc - vb) @ n
            if vrel < 0:
                rxn = r[0] * n[1] - r[1] * n[0]
                j = -(1 + BOX_RESTITUTION) * vrel / (1 / CAR_MASS + 1 / mb + rxn ** 2 / I)
                vc = vc + j / CAR_MASS * n
                self.vel[k] -= j / mb * n
                self.angv[k] -= rxn * j / I
                car.speed = float(vc @ hdir)                       # 차는 옆으로 미끄러지지 않음
        return hit

    # ---------------------------------------------------------------- 한 스텝
    def step(self, dt, track, cones=None):
        if len(self) == 0:
            return
        # 이동 + 마찰
        self.pos += self.vel * dt
        self.ang += self.angv * dt
        sp = np.linalg.norm(self.vel, axis=1, keepdims=True)
        new = np.maximum(0.0, sp * np.exp(-BOX_DAMP * dt) - BOX_DECEL * dt)
        self.vel *= np.where(sp > 1e-9, new / np.maximum(sp, 1e-9), 0.0)
        self.angv *= np.exp(-ANG_DAMP * dt)
        # 박스끼리 (원으로 근사)
        n_box = len(self)
        for a in range(n_box):
            for b in range(a + 1, n_box):
                d = self.pos[a] - self.pos[b]
                dist = np.hypot(*d)
                lim = (self.size[a] + self.size[b]) * 0.5
                if dist < lim and dist > 1e-6:
                    n = d / dist
                    ma, mb = self.mass[a], self.mass[b]
                    pen = lim - dist
                    self.pos[a] += n * pen * mb / (ma + mb)
                    self.pos[b] -= n * pen * ma / (ma + mb)
                    vr = (self.vel[a] - self.vel[b]) @ n
                    if vr < 0:
                        j = -(1.3) * vr / (1 / ma + 1 / mb)
                        self.vel[a] += j / ma * n
                        self.vel[b] -= j / mb * n
        # 박스-콘 (콘은 고정)
        if cones is not None and len(cones):
            for k in range(n_box):
                d = self.pos[k] - cones[:, :2]
                dist = np.hypot(d[:, 0], d[:, 1])
                lim = cones[:, 2] + self.size[k] * 0.5
                for m in np.flatnonzero(dist < lim):
                    n = d[m] / max(dist[m], 1e-6)
                    self.pos[k] += n * (lim[m] - dist[m])
                    vn = self.vel[k] @ n
                    if vn < 0:
                        self.vel[k] -= (1 + WALL_RESTITUTION) * vn * n
        # 박스-벽: 모서리가 트랙 밖으로 나가면 안쪽으로 밀어 넣음
        if self.idx is None:
            self.idx = [track.nearest_idx(p) for p in self.pos]
        corners = self.corners()
        for k in range(n_box):
            self.idx[k], _, _ = track.project(self.pos[k], self.idx[k], back=12, ahead=12)
            for c in corners[k]:
                i, s, dist = track.project(c, self.idx[k], back=12, ahead=12)
                out = dist - track.half[i]
                if out > 0:
                    p = track.point_at(s)
                    n = (c - p) / max(dist, 1e-6)                  # 바깥 방향
                    self.pos[k] -= n * out
                    vn = self.vel[k] @ n
                    if vn > 0:
                        self.vel[k] -= (1 + WALL_RESTITUTION) * vn * n
                        self.angv[k] *= 0.5
                    corners = self.corners()


class Shuttles:
    def __init__(self, items, track):
        self.items = items
        self.base = np.array([track.center[i] for i, *_ in items]).reshape(-1, 2)
        self.dir = np.array([track.normal[i] for i, *_ in items]).reshape(-1, 2)
        a = np.array([it[1:] for it in items], float).reshape(-1, 4)
        self.amp, self.period, self.phase, self.r = a.T
        self.update(0.0)

    def __len__(self):
        return len(self.items)

    def update(self, t):
        w = 2 * np.pi / np.maximum(self.period, 1e-6)
        arg = w * t + 2 * np.pi * self.phase
        self.pos = self.base + self.dir * (self.amp * np.sin(arg))[:, None]
        self.vel = self.dir * (self.amp * w * np.cos(arg))[:, None]

    def circles(self):
        return np.concatenate([self.pos, self.r[:, None]], axis=1) if len(self) else np.zeros((0, 3))

    def collide_car(self, car, car_r, bounce):
        if len(self) == 0:
            return False
        d = car.pos - self.pos
        dist = np.hypot(d[:, 0], d[:, 1])
        lim = self.r + car_r
        hit = np.flatnonzero(dist < lim)
        for k in hit:
            n = d[k] / max(dist[k], 1e-6)
            car.x, car.y = self.pos[k] + n * lim[k]
            hdir = np.array([np.cos(car.heading), np.sin(car.heading)])
            if car.speed * (hdir @ n) < 0:                         # 차가 셔틀 쪽으로 가는 중
                car.speed *= bounce
        return len(hit) > 0
