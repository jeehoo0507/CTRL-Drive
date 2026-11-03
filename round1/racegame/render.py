"""렌더링: 학습할 땐 안 쓰고, 영상/리플레이 만들 때만 쓴다 (PIL만 필요)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .car import CAR_LENGTH, CAR_WIDTH, MAX_SPEED
from .sensors import RAY_ANGLES, RAY_MAX

CANVAS = (1008, 704)       # 16의 배수 → mp4 인코딩에 안전
OFFSET = (4, 2)
GRASS = (46, 110, 64)
ROAD = (72, 74, 80)
WALL = (235, 235, 235)
TRAIL = (255, 200, 60)
CAR = (230, 57, 70)
RAY = (120, 220, 255)


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _pts(arr):
    return [(float(x) + OFFSET[0], float(y) + OFFSET[1]) for x, y in arr]


def _area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _background(track):
    if getattr(track, "_bg", None) is not None:      # 트랙 객체에 직접 캐시
        return track._bg
    img = Image.new("RGB", CANVAS, GRASS)
    d = ImageDraw.Draw(img)
    outer, inner = (track.left, track.right) if _area(track.left) > _area(track.right) \
        else (track.right, track.left)
    d.polygon(_pts(outer), fill=ROAD)
    d.polygon(_pts(inner), fill=GRASS)
    d.line(_pts(np.vstack([track.left, track.left[:1]])), fill=WALL, width=3)
    d.line(_pts(np.vstack([track.right, track.right[:1]])), fill=WALL, width=3)
    # 체크무늬 출발선
    c, t = track.center[0], track.tangent[0]
    n = np.array([-t[1], t[0]])
    half = track.half[0]
    k = max(8, int(round(2 * half / 11)))
    w = 2 * half / k
    for i in range(k):
        for j in range(2):
            a = c + n * (-half + w * i) + t * (j * 6 - 6)
            col = (255, 255, 255) if (i + j) % 2 == 0 else (20, 20, 20)
            d.polygon(_pts([a, a + n * w, a + n * w + t * 6, a + t * 6]), fill=col)
    d.text((14, CANVAS[1] - 30), f"{track.name}  ·  {track.category}",
           fill=(220, 240, 220), font=_font(18))
    track._bg = img
    return img


def draw_obstacles(d, obstacles):
    """장애물 = 주황 콘 (위에서 본 모습)."""
    for x, y, r in (obstacles if obstacles is not None else []):
        x, y = x + OFFSET[0], y + OFFSET[1]
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 120, 20), outline=(60, 25, 0), width=2)
        d.ellipse([x - r * 0.62, y - r * 0.62, x + r * 0.62, y + r * 0.62], fill=(250, 245, 240))
        d.ellipse([x - r * 0.35, y - r * 0.35, x + r * 0.35, y + r * 0.35], fill=(255, 120, 20))


def box_color(mass):
    """가벼울수록 밝은 나무색, 무거울수록 어두운 강철색."""
    t = float(np.clip((np.log(mass) - np.log(0.4)) / (np.log(3.0) - np.log(0.4)), 0, 1))
    wood, steel = np.array([214, 160, 92]), np.array([92, 100, 112])
    return tuple(int(c) for c in wood * (1 - t) + steel * t)


def draw_boxes(d, corners, masses, outline=(40, 25, 10)):
    for quad, m in zip(corners, masses):
        pts = _pts(quad)
        d.polygon(pts, fill=box_color(m), outline=outline, width=2)
        d.line([pts[0], pts[2]], fill=outline, width=1)            # 나무 상자 X 무늬
        d.line([pts[1], pts[3]], fill=outline, width=1)


def draw_shuttles(d, sh, rails=True):
    if sh is None or len(sh) == 0:
        return
    if rails:
        for b, u, a in zip(sh.base, sh.dir, sh.amp):
            d.line(_pts([b - u * a, b + u * a]), fill=(150, 110, 190), width=2)
    for (x, y), r in zip(sh.pos, sh.r):
        x, y = x + OFFSET[0], y + OFFSET[1]
        d.ellipse([x - r, y - r, x + r, y + r], fill=(170, 60, 220), outline=(40, 0, 60), width=2)
        d.ellipse([x - r * 0.4, y - r * 0.4, x + r * 0.4, y + r * 0.4], fill=(240, 200, 255))


def draw_pickups(d, strat):
    """⚡ 에너지 = 노란 다이아몬드, 🔧 타이어 = 하늘색 링."""
    if strat is None or not strat.on or not len(strat.items):
        return
    for (x, y, kind), alive in zip(strat.items, strat.alive):
        if not alive:
            continue
        x, y = x + OFFSET[0], y + OFFSET[1]
        if kind == 0.5:                              # 코인
            r = 8
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 200, 0), outline=(130, 80, 0), width=2)
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 240, 150))
        elif kind > 0:
            r = 9
            d.polygon([(x, y - r), (x + r * 0.75, y), (x, y + r), (x - r * 0.75, y)],
                      fill=(255, 225, 40), outline=(90, 70, 0))
            d.line([(x + 2, y - 6), (x - 3, y + 1), (x + 3, y - 1), (x - 2, y + 6)], fill=(90, 70, 0), width=2)
        else:
            r = 9
            d.ellipse([x - r, y - r, x + r, y + r], fill=(70, 210, 240), outline=(0, 60, 80), width=2)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(15, 18, 24))


def draw_flame(d, car):
    c, s = np.cos(car.heading), np.sin(car.heading)
    hl = CAR_LENGTH / 2
    tip = (car.x - c * (hl + 16), car.y - s * (hl + 16))
    l = (car.x - c * hl - s * 5, car.y - s * hl + c * 5)
    r = (car.x - c * hl + s * 5, car.y - s * hl - c * 5)
    d.polygon(_pts([l, tip, r]), fill=(255, 140, 0))
    tip2 = (car.x - c * (hl + 8), car.y - s * (hl + 8))
    d.polygon(_pts([l, tip2, r]), fill=(255, 240, 120))


def draw_strategy_hud(d, strat, y0):
    """에너지 / 타이어 게이지."""
    fs = _font(14)
    for k, (label, val, col) in enumerate([("energy", strat.energy, (255, 210, 40)),
                                           ("tire", strat.grip(), (90, 220, 120) if strat.grip() > 0.8
                                            else (240, 170, 60) if strat.grip() > 0.65 else (240, 80, 80))]):
        y = y0 + 20 * k
        d.text((22, y), label, fill=(200, 200, 210), font=fs)
        bar = (90, y + 4, 258, y + 14)
        d.rectangle(bar, outline=(90, 90, 100))
        d.rectangle((bar[0], bar[1], bar[0] + (bar[2] - bar[0]) * float(np.clip(val, 0, 1)), bar[3]),
                    fill=col)


def draw_frame(env, title=None):
    car, track = env.car, env.track
    if not hasattr(env, "_trail") or env.steps == 0:
        env._trail = []
    env._trail.append((car.x, car.y))

    img = _background(track).copy()
    d = ImageDraw.Draw(img)
    draw_obstacles(d, getattr(env, "obstacles", None))
    draw_shuttles(d, getattr(env, "shuttles", None))
    if getattr(env, "boxes", None) is not None and len(env.boxes):
        draw_boxes(d, env.boxes.corners(), env.boxes.mass)
    strat = getattr(env, "strat", None)
    draw_pickups(d, strat)

    if len(env._trail) > 1:
        d.line(_pts(env._trail), fill=TRAIL, width=2)

    # 레이캐스트
    rays = env.last_obs[: len(RAY_ANGLES)] * RAY_MAX
    for ang, r in zip(RAY_ANGLES, rays):
        a = car.heading + ang
        end = (car.x + np.cos(a) * r, car.y + np.sin(a) * r)
        d.line(_pts([(car.x, car.y), end]), fill=RAY, width=1)
        ex, ey = _pts([end])[0]
        d.ellipse([ex - 2, ey - 2, ex + 2, ey + 2], fill=RAY)

    # 차 (회전된 직사각형)
    if strat is not None and strat.boosting:
        draw_flame(d, car)
    c, s = np.cos(car.heading), np.sin(car.heading)
    hl, hw = CAR_LENGTH / 2, CAR_WIDTH / 2
    corners = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    body = [(car.x + x * c - y * s, car.y + x * s + y * c) for x, y in corners]
    d.polygon(_pts(body), fill=CAR, outline=(40, 10, 10))
    wx = [(hl * 0.55, hw * 0.8), (hl * 0.55, -hw * 0.8), (hl * 0.1, -hw * 0.8), (hl * 0.1, hw * 0.8)]
    d.polygon(_pts([(car.x + x * c - y * s, car.y + x * s + y * c) for x, y in wx]),
              fill=(40, 40, 60))

    # HUD
    info = env.last_info or {"progress": 0.0, "time": 0.0}
    on = strat is not None and strat.on
    panel = [10, 10, 270, 162 if on else 118]
    d.rounded_rectangle(panel, radius=10, fill=(15, 18, 24))
    f, fs = _font(20), _font(15)
    d.text((22, 18), title or "AI RACE", fill=(255, 255, 255), font=f)
    d.text((22, 46), f"time   {info['time']:6.2f} s", fill=(200, 200, 210), font=fs)
    d.text((22, 66), f"speed  {car.speed / MAX_SPEED * 100:5.0f} %", fill=(200, 200, 210), font=fs)
    lap_txt = f"lap {min(int(env.best_progress) + 1, strat.laps)}/{strat.laps}" if on and strat.laps > 1 \
        else f"lap    {info['progress'] * 100:5.1f} %"
    d.text((22, 86), lap_txt, fill=(200, 200, 210), font=fs)
    bar = (150, 90, 258, 100)
    d.rectangle(bar, outline=(90, 90, 100))
    d.rectangle((bar[0], bar[1], bar[0] + (bar[2] - bar[0]) * info["progress"], bar[3]),
                fill=TRAIL)
    if on:
        draw_strategy_hud(d, strat, 112)
    status = "FINISH!" if info.get("finished") else "CRASH" if info.get("crashed") \
        else "STUCK" if info.get("stuck") else None
    if status:
        col = (90, 220, 120) if status == "FINISH!" else (240, 80, 80)
        d.text((150, 46), status, fill=col, font=_font(22))
    return img
