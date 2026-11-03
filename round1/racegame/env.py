"""env: 한 판의 규칙. track + car + sensors 를 묶는다.

    env = RaceEnv("oval")
    obs = env.reset(seed=0)
    obs, reward, done, info = env.step([steer, throttle])

- reward : 학습용 기본 보상 (참가자가 마음대로 바꿔도 됨)
- info   : 공식 측정값 (채점은 오직 이것만 본다)
"""
import os
import numpy as np

from .track import Track
from .car import Car
from . import sensors
from .dynamic import Boxes, Shuttles
from .strategy import Strategy

TRACK_DIR = os.path.join(os.path.dirname(__file__), "tracks")
DT = 0.05             # 20 FPS 고정 timestep → 결정론적
MAX_STEPS = 1200      # 60초
STUCK_STEPS = 60      # 3초 동안 전진이 없으면 종료
CAR_RADIUS = 9.0      # 장애물 충돌 판정용
BOUNCE = -0.25        # 장애물에 박으면 속도가 반대로 25% 튕김 (탈락 X, 막히기만)
PAR_SPEED = 230.0     # 트랙 json에 par_time(기준기록)이 없을 때: 중심선 길이 / 230px/s


def resolve_track(name_or_path):
    """'oval' 같은 이름이면 내장 트랙, json 경로면 그 파일."""
    if name_or_path.endswith(".json") and os.path.exists(name_or_path):
        return name_or_path
    return os.path.join(TRACK_DIR, name_or_path, "track.json")


def list_tracks():
    return sorted(d for d in os.listdir(TRACK_DIR)
                  if os.path.exists(os.path.join(TRACK_DIR, d, "track.json")))


class RaceEnv:
    def __init__(self, track, max_steps=MAX_STEPS):
        if isinstance(track, Track):
            self.track = track
        else:
            self.track = Track(track if isinstance(track, dict) else resolve_track(track))
        self.strat = Strategy(self.track.raw.get("strategy"))   # 전략 모드 (없으면 꺼짐)
        self.max_steps = max_steps * self.strat.laps
        par = self.track.raw.get("par_time")
        self.par_time = float(par) if par else \
            round(self.track.total_length * self.strat.laps / PAR_SPEED, 2)
        self.car = None

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        x, y, h = self.track.start_pose
        # 시드에 따라 출발 위치/방향을 살짝 흔든다 → 시드 외우기 방지
        side = rng.uniform(-0.15, 0.15) * 2 * min(self.track.half[0], 45.0)
        x -= np.sin(h) * side
        y += np.cos(h) * side
        h += rng.uniform(-0.08, 0.08)
        self.car = Car(x, y, h)
        # 장애물은 별도 난수 흐름 (출발 위치 흔들기와 독립)
        self.obstacles = self.track.make_obstacles(
            np.random.default_rng(None if seed is None else seed + 7919))
        # 움직이는 장애물 (밀 수 있는 박스, 왕복 셔틀) — 각각 별도 난수 흐름
        sd = (lambda k: None if seed is None else seed + k)
        self.boxes = Boxes(self.track.make_boxes(np.random.default_rng(sd(104729))))
        self.shuttles = Shuttles(self.track.make_shuttles(np.random.default_rng(sd(15485863))),
                                 self.track)
        self.t = 0.0
        self.hits = 0
        self._touching = False
        self.strat.reset(self.track, np.random.default_rng(sd(32452843)))

        self.idx, self.s_prev, _ = self.track.project(self.car.pos, 0, back=10, ahead=10)
        self.dist = 0.0              # 출발선부터 누적 주행 거리 (역주행하면 줄어듦)
        self.best_progress = 0.0
        self.last_improve = 0
        self.steps = 0
        self.done = False
        self.last_info = {}
        return self._obs()

    def step(self, action):
        assert not self.done, "에피소드가 끝났습니다. reset()을 호출하세요."
        a = np.asarray(action, dtype=float).reshape(-1)
        boost, grip = self.strat.controls(a, DT)
        self.car.update(a[0], a[1], DT, boost=boost, grip=grip)
        self.strat.after_move(self.car, DT)
        self.steps += 1
        self.t += DT
        self.shuttles.update(self.t)
        self.boxes.step(DT, self.track, self.obstacles)
        contact = self._collide_obstacles()
        contact |= self.boxes.collide_car(self.car, CAR_RADIUS)
        contact |= self.shuttles.collide_car(self.car, CAR_RADIUS, BOUNCE)
        if contact and not self._touching:     # 붙어서 비비는 동안은 1회로 셈
            self.hits += 1
        self._touching = contact

        # 진행도 갱신
        self.idx, s, off = self.track.project(self.car.pos, self.idx)
        if self.strat.soft_wall(self, s, off, self.track.half[self.idx]):   # 소프트 벽
            self.idx, s, off = self.track.project(self.car.pos, self.idx)
        L = self.track.total_length
        delta = s - self.s_prev
        if delta < -L / 2:
            delta += L
        elif delta > L / 2:
            delta -= L
        self.s_prev = s
        self.dist += delta
        progress = self.dist / L

        reward = 100.0 * delta / L          # 한 바퀴 = 100
        if progress > self.best_progress + 1e-4:
            self.best_progress = progress
            self.last_improve = self.steps

        self.strat.pickups(self.car, self.best_progress, CAR_RADIUS)
        crashed = off > self.track.half[self.idx] + (1e9 if self.strat.soft else 0)
        finished = self.best_progress >= self.strat.laps
        stuck = self.steps - self.last_improve > STUCK_STEPS
        timeout = self.steps >= self.max_steps
        if crashed:
            reward -= 10.0
        self.done = crashed or finished or stuck or timeout

        self.last_info = {
            "track": self.track.name,
            "category": self.track.category,
            "progress": float(min(self.best_progress / self.strat.laps, 1.0)),
            "finished": bool(finished),
            "crashed": bool(crashed),
            "stuck": bool(stuck and not crashed and not finished),
            "time": round(self.steps * DT, 2),
            "steps": self.steps,
            "hits": self.hits,                 # 장애물 충돌 횟수 (참고용, 감점 없음)
            "par_time": self.par_time,         # 이 트랙 기준기록 (채점용)
            **self.strat.info(),
        }
        return self._obs(), float(reward), self.done, self.last_info

    # ------------------------------------------------------------------
    def _collide_obstacles(self):
        """장애물에 박으면 밖으로 밀어내고 튕김. 탈락은 아님."""
        if len(self.obstacles) == 0:
            return False
        c = self.obstacles
        dx, dy = self.car.x - c[:, 0], self.car.y - c[:, 1]
        d = np.hypot(dx, dy)
        lim = c[:, 2] + CAR_RADIUS
        hit = np.flatnonzero(d < lim)
        if len(hit) == 0:
            return False
        for k in hit:
            dk = max(d[k], 1e-6)
            self.car.x = c[k, 0] + dx[k] / dk * lim[k]
            self.car.y = c[k, 1] + dy[k] / dk * lim[k]
        self.car.speed *= BOUNCE
        return True

    def _obs(self):
        circles = self.obstacles
        if len(self.shuttles):
            circles = np.vstack([circles, self.shuttles.circles()])
        segs = self.boxes.segments() if len(self.boxes) else None
        base = sensors.observe(self.car, self.track, self.s_prev, circles, segs)
        # 전략 모드 트랙에서만 관측이 19개로 늘어남 (Round 1 트랙은 전부 12개)
        self.last_obs = np.concatenate([base, self.strat.obs(self.car, self.best_progress)]) \
            if self.strat.on else base
        return self.last_obs

    def render(self, title=None):
        """현재 화면을 PIL 이미지로 (영상 저장용)."""
        from .render import draw_frame
        return draw_frame(self, title)
