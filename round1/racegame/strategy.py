"""전략 모드: 정답이 하나가 아니게 만드는 장치들. 트랙 json의 "strategy" 블록으로 켬.

    "strategy": {
        "laps": 2,                  # 여러 바퀴 → 페이스 조절이 의미 있어짐
        "boost": true,              # 행동 3번째 값 > 0 이면 니트로 (에너지 소모)
        "energy_start": 0.6,        # 시작 에너지 (0~1)
        "tire_wear": 0.1,           # 코너를 세게 돌수록 타이어가 닳아 그립 감소
        "soft_walls": true,         # 벽에 닿으면 탈락 대신 감속 + 타이어 손상
        "pickups": {"energy": 6, "tire": 3, "coin": 5},  # 바퀴마다 새로 깔리는 아이템
        "coin_bonus": 0.5,           # 코인 1개당 기록에서 빼주는 시간(초)
        "pickup_place": "anywhere",  # "edge" 면 벽 쪽에만 → 일부러 가야 주움
        "energy_regen": 0.0,         # 에너지 자동 충전 (/s). 크면 '아끼면 손해'
        "pickup_value": 1.0          # 아이템 효과 배수. 크면 '주우러 가는 게 이득'
    }

트레이드오프
  부스트   : 빠르지만 그립이 떨어지고 에너지를 씀 → 직선에서 쓸까, 아껴 둘까
  타이어   : 세게 달리면 다음 바퀴가 느려짐 → 1랩 전력 vs 2랩 관리
  아이템   : 주우려면 라인을 벗어나야 함 → 돌아가서 주울 가치가 있나
  소프트 벽: 벽을 긁어도 안 죽음 → 위험을 감수하고 붙어 달릴까
"""
import numpy as np

from .car import LAT_GRIP

BOOST_DRAIN = 0.30      # 부스트 중 에너지 소모 (/s) → 가득 차 있으면 약 3.3초
GRIP_LOSS = 0.45        # 타이어가 완전히 닳으면 그립 55%
WALL_SLOW = 0.45        # 벽에 닿으면 속도 45%로
WALL_WEAR = 0.06        # 벽에 닿을 때마다 타이어 마모
ENERGY_GAIN = 0.35
TIRE_GAIN = 0.40
PICKUP_R = 11.0
SEE_DIST = 300.0
EXTRA_DIM = 7           # obs[12:19]


class Strategy:
    def __init__(self, spec):
        self.on = spec is not None
        spec = spec or {}
        self.laps = int(spec.get("laps", 1))
        self.boost_on = bool(spec.get("boost", False))
        self.energy0 = float(spec.get("energy_start", 0.6 if self.boost_on else 0.0))
        self.wear_k = float(spec.get("tire_wear", 0.0))
        self.soft = bool(spec.get("soft_walls", False))
        self.pick_spec = spec.get("pickups") or {}
        self.pick_place = spec.get("pickup_place", "anywhere")
        self.regen = float(spec.get("energy_regen", 0.0))
        self.pick_value = float(spec.get("pickup_value", 1.0))
        self.coin_bonus = float(spec.get("coin_bonus", 0.5))

    # ------------------------------------------------------------------
    def reset(self, track, rng):
        self.track, self.rng = track, rng
        self.energy, self.wear = self.energy0, 0.0
        self.boosting = False
        self.boost_time = 0.0
        self.wall_hits = 0
        self._on_wall = False
        self.picked = {"energy": 0, "tire": 0, "coin": 0}
        self.lap = 0
        self._spawn()

    def _spawn(self):
        """바퀴마다 아이템을 트랙 폭 전체에 흩뿌림 (라인 위에 있을 수도, 벽 끝에 있을 수도)."""
        tr, rng, items = self.track, self.rng, []
        for kind, code in (("energy", 1.0), ("tire", -1.0), ("coin", 0.5)):
            for f in rng.uniform(0.05, 0.97, int(self.pick_spec.get(kind, 0))):
                i = int(f * tr.n) % tr.n
                room = max(0.0, tr.half[i] - 10)
                if self.pick_place == "edge":                   # 벽 쪽 (좌우 랜덤) → 라인에서 벗어나야 함
                    lat = rng.choice([-1.0, 1.0]) * room * rng.uniform(0.6, 0.8)
                else:
                    lat = rng.uniform(-1, 1) * room
                x, y = tr.center[i] + tr.normal[i] * lat
                items.append((x, y, code))
        self.items = np.array(items, float).reshape(-1, 3)
        self.alive = np.ones(len(self.items), bool)

    # ------------------------------------------------------------------
    def grip(self):
        return 1.0 - GRIP_LOSS * self.wear

    def controls(self, action, dt):
        """→ (부스트 여부, 그립 배수). 일반 트랙에선 항상 (False, 1.0)."""
        if not self.on:
            return False, 1.0
        boost = self.boost_on and len(action) > 2 and action[2] > 0 and self.energy > 0
        if boost:
            self.energy = max(0.0, self.energy - BOOST_DRAIN * dt)
            self.boost_time += dt
        elif self.regen > 0:
            self.energy = min(1.0, self.energy + self.regen * dt)
        self.boosting = bool(boost)
        return self.boosting, self.grip()

    def after_move(self, car, dt):
        if self.on and self.wear_k > 0:
            lat = abs(car.yaw_rate * car.speed) / LAT_GRIP           # 횡가속 사용률 (0~1)
            self.wear = min(1.0, self.wear + self.wear_k * lat ** 2 * dt * (1.5 if self.boosting else 1.0))

    def soft_wall(self, env, s, off, half):
        """벽 밖으로 나가면 안쪽으로 밀어 넣고 감속. 처리했으면 True."""
        if not (self.on and self.soft):
            return False
        if off <= half:
            self._on_wall = False
            return False
        car = env.car
        p = self.track.point_at(s)
        n = (car.pos - p) / max(off, 1e-6)
        car.x, car.y = p + n * (half - 1.0)
        car.speed *= WALL_SLOW
        if not self._on_wall:
            self.wall_hits += 1
            if self.wear_k > 0:
                self.wear = min(1.0, self.wear + WALL_WEAR)
        self._on_wall = True
        return True

    def pickups(self, car, best_progress, car_r):
        if not self.on:
            return
        lap = int(best_progress)
        if lap > self.lap and lap < self.laps:      # 새 바퀴 → 아이템 다시 깔기
            self.lap = lap
            self._spawn()
        if not len(self.items):
            return
        d = np.hypot(self.items[:, 0] - car.x, self.items[:, 1] - car.y)
        for k in np.flatnonzero(self.alive & (d < PICKUP_R + car_r)):
            self.alive[k] = False
            if self.items[k, 2] == 0.5:
                self.picked["coin"] += 1
            elif self.items[k, 2] > 0:
                self.energy = min(1.0, self.energy + ENERGY_GAIN * self.pick_value)
                self.picked["energy"] += 1
            else:
                self.wear = max(0.0, self.wear - TIRE_GAIN * self.pick_value)
                self.picked["tire"] += 1

    # ------------------------------------------------------------------
    def obs(self, car, best_progress):
        """obs[12:19] = 에너지, 그립, 부스트중, 전체 진행률, 가장 가까운 아이템(거리, 각도, 종류)
        종류: +1 에너지, -1 타이어, +0.5 코인, 0 없음"""
        dist, ang, kind = 1.0, 0.0, 0.0
        if self.on and len(self.items) and self.alive.any():
            it = self.items[self.alive]
            dx, dy = it[:, 0] - car.x, it[:, 1] - car.y
            d = np.hypot(dx, dy)
            a = (np.arctan2(dy, dx) - car.heading + np.pi) % (2 * np.pi) - np.pi
            ok = (d < SEE_DIST) & (np.abs(a) < np.radians(100))      # 앞쪽 시야 안에 있는 것만
            if ok.any():
                k = np.flatnonzero(ok)[np.argmin(d[ok])]
                dist, ang, kind = d[k] / SEE_DIST, a[k] / np.pi, it[k, 2]
        return np.array([self.energy if self.on else 0.0, self.grip(), float(self.boosting),
                         min(best_progress / self.laps, 1.0), dist, ang, kind], np.float32)

    def info(self):
        if not self.on:
            return {}
        return {"laps": self.laps, "energy": round(self.energy, 3), "tire_wear": round(self.wear, 3),
                "coin_time": round(self.picked["coin"] * self.coin_bonus, 2),
                "boost_time": round(self.boost_time, 2), "wall_hits": self.wall_hits,
                "picked": dict(self.picked)}
