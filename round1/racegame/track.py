"""트랙: 어디가 길인지.

트랙 json에는 '제어점(control_points)'과 '폭(width)'만 적는다.
여기서 Catmull-Rom 스플라인으로 부드러운 중심선을 만들고,
그 중심선 양옆으로 벽(선분 목록)을 만든다.
"""
import json
import numpy as np

SPACING = 8.0  # 중심선 점 간격(px)


def _catmull_rom_closed(ctrl, samples_per_seg=40):
    """닫힌 곡선용 Catmull-Rom 스플라인."""
    ctrl = np.asarray(ctrl, dtype=float)
    n = len(ctrl)
    pts = []
    for i in range(n):
        p0, p1, p2, p3 = ctrl[(i - 1) % n], ctrl[i], ctrl[(i + 1) % n], ctrl[(i + 2) % n]
        for t in np.linspace(0, 1, samples_per_seg, endpoint=False):
            t2, t3 = t * t, t * t * t
            pts.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(pts)


def _resample_closed(pts, spacing):
    """닫힌 폴리라인을 일정 간격으로 다시 샘플링."""
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    n = max(int(total // spacing), 8)
    targets = np.linspace(0, total, n, endpoint=False)
    x = np.interp(targets, s, closed[:, 0])
    y = np.interp(targets, s, closed[:, 1])
    return np.stack([x, y], axis=1)


def _radius(pts, k=2):
    """각 점의 곡률 반경 (k칸 떨어진 세 점으로 원을 그려 추정)."""
    a, b, c = np.roll(pts, k, axis=0), pts, np.roll(pts, -k, axis=0)
    ab, bc, ca = (np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                  np.linalg.norm(a - c, axis=1))
    cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    return ab * bc * ca / np.maximum(2 * cross, 1e-9)


def _smooth_until(pts, min_r, max_iter=300, spread=6):
    """급커브 때문에 안쪽 벽이 꼬이지 않도록, 반경이 min_r 미만인 '그 부분만' 둥글게."""
    for _ in range(max_iter):
        bad = _radius(pts) < min_r
        if not bad.any():
            break
        for s in range(1, spread + 1):            # 주변 점까지 같이 펴기
            bad = bad | np.roll(bad, s) | np.roll(bad, -s)
        smooth = (np.roll(pts, 1, axis=0) + 2 * pts + np.roll(pts, -1, axis=0)) / 4
        pts = np.where(bad[:, None], smooth, pts)
    return pts


class Track:
    def __init__(self, src):
        """src: track.json 경로 또는 같은 내용의 dict (랜덤 트랙 생성기용)."""
        if isinstance(src, dict):
            data = src
        else:
            with open(src, encoding="utf-8") as f:
                data = json.load(f)
        self.raw = data
        self.name = data["name"]
        self.category = data.get("category", "general")
        self.size = tuple(data.get("size", [1000, 700]))  # 렌더링 캔버스 크기
        # 폭: 고정("width") 또는 구간별로 변하는 폭("width_profile": [[트랙 비율 0~1, 폭], ...])
        profile = data.get("width_profile")
        width_at = (lambda f: np.interp(f, *zip(*profile), period=1.0)) if profile \
            else (lambda f: np.full_like(f, float(data["width"]), dtype=float))

        raw = _catmull_rom_closed(data["control_points"])
        center = _resample_closed(raw, SPACING)
        # 최소 커브 반경 (기본 폭 x 0.65). 날카로운 트랙은 json에서 낮출 수 있음 (>= 0.55 권장)
        frac0 = np.arange(len(center)) / len(center)
        center = _smooth_until(center, width_at(frac0) * data.get("min_radius_ratio", 0.65),
                               max_iter=data.get("smooth_iters", 300),
                               spread=data.get("smooth_spread", 6))
        self.center = _resample_closed(center, SPACING)          # (N, 2)
        self.half = width_at(np.arange(len(self.center)) / len(self.center)) / 2   # 점마다 반폭
        self.width = float(2 * self.half.min())                  # (호환용) 가장 좁은 폭
        self.n = len(self.center)
        nxt = np.roll(self.center, -1, axis=0)
        self.seg_vec = nxt - self.center                          # 각 구간 벡터
        self.seg_len = np.linalg.norm(self.seg_vec, axis=1)
        self.tangent = self.seg_vec / self.seg_len[:, None]
        self.s_start = np.concatenate([[0], np.cumsum(self.seg_len)[:-1]])  # 각 구간 시작 거리
        self.radius = np.minimum(_radius(self.center, 2), _radius(self.center, 4))  # 각 점 커브 반경
        self.total_length = float(self.seg_len.sum())

        # 벽: 각 점의 법선 방향으로 width/2 만큼 밀어낸 두 폴리라인
        prev_t = np.roll(self.tangent, 1, axis=0)
        avg_t = self.tangent + prev_t
        avg_t /= np.linalg.norm(avg_t, axis=1, keepdims=True)
        normal = np.stack([-avg_t[:, 1], avg_t[:, 0]], axis=1)
        self.normal = normal
        self.left = self.center + normal * self.half[:, None]
        self.right = self.center - normal * self.half[:, None]
        a = np.vstack([self.left, self.right])
        b = np.vstack([np.roll(self.left, -1, axis=0), np.roll(self.right, -1, axis=0)])
        self.wall_a, self.wall_b = a, b                           # (2N, 2) 선분 시작/끝

        t0 = self.tangent[0]
        self.start_pose = (*self.center[0], float(np.arctan2(t0[1], t0[0])))

        # 고정 장애물: "obstacles": [[트랙 비율 0~1, 좌우 위치 -1~1(반폭 기준), 반지름], ...]
        self.fixed_obstacles = np.array([self.obstacle_xy(f, lat, r)
                                         for f, lat, r in data.get("obstacles", [])]).reshape(-1, 3)
        # 랜덤 장애물: 시드마다 위치가 바뀜 → 트랙 암기 불가
        #   "random_obstacles": {"count": 8, "r": [10, 16], "clear_start": 0.08}
        self.random_obstacles = data.get("random_obstacles")

    def obstacle_xy(self, frac, lat, r):
        i = int(frac * self.n) % self.n
        p = self.center[i] + self.normal[i] * lat * self.half[i]
        return (float(p[0]), float(p[1]), float(r))

    def make_obstacles(self, rng):
        """고정 장애물 + (설정돼 있으면) 시드 기반 랜덤 장애물. 반환 (M, 3) [x, y, r]"""
        obs = [self.fixed_obstacles]
        spec = self.random_obstacles
        if spec:
            n, (rmin, rmax) = spec["count"], spec.get("r", [10, 16])
            clear = spec.get("clear_start", 0.08)
            min_gap = spec.get("min_gap", 34.0)           # 차가 빠져나갈 최소 틈 (차 폭 12)
            fracs = np.sort(rng.uniform(clear, 1 - 0.03, n))
            out = []
            for f in fracs:
                i = int(f * self.n) % self.n
                h = self.half[i]
                r = rng.uniform(rmin, rmax)
                if 2 * (h - r) < min_gap:                 # 너무 좁은 곳엔 안 놓음
                    continue
                y = rng.uniform(-(h - r), h - r)          # 중심선 기준 좌우 위치(px)
                gap_l, gap_r = h - (y + r), h + (y - r)
                if max(gap_l, gap_r) < min_gap:           # 양쪽 다 막히면 한쪽 벽에 붙임
                    y = (h - r) * rng.choice([-1.0, 1.0])
                out.append(self.obstacle_xy(f, y / h, r))
            obs.append(np.array(out).reshape(-1, 3))
        return np.vstack(obs) if len(obs) else np.zeros((0, 3))

    # ------------------------------------------------------------------ 움직이는 장애물
    def _random_slots(self, rng, n, size_rng, clear=0.08, min_gap=34.0):
        """랜덤 위치 n개: (트랙비율, 좌우 px, 크기). 한쪽에는 항상 min_gap 이상 틈을 남김."""
        out = []
        for f in np.sort(rng.uniform(clear, 0.97, n)):
            i = int(f * self.n) % self.n
            h = self.half[i]
            size = rng.uniform(*size_rng)
            r = size * 0.72                                 # 박스 대각선 반쯤 (회전 고려)
            if 2 * (h - r) < min_gap:
                continue
            y = rng.uniform(-(h - r), h - r)
            if max(h - (y + r), h + (y - r)) < min_gap:
                y = (h - r) * rng.choice([-1.0, 1.0])
            out.append((f, y, size))
        return out

    def make_boxes(self, rng):
        """밀 수 있는 박스. 반환 [(x, y, 한 변, 질량, 각도), ...]
        고정: "boxes": [[트랙비율, 좌우 -1~1, 한 변, 질량], ...]
        랜덤: "random_boxes": {"count": 8, "size": [22, 34], "mass": [0.4, 3.0]}"""
        items = []
        for f, lat, size, mass in self.raw.get("boxes", []):
            x, y, _ = self.obstacle_xy(f, lat, 0)
            items.append((x, y, size, mass, rng.uniform(0, np.pi / 2)))
        spec = self.raw.get("random_boxes")
        if spec:
            for f, ypx, size in self._random_slots(rng, spec["count"], spec.get("size", [22, 34]),
                                                   spec.get("clear_start", 0.08)):
                i = int(f * self.n) % self.n
                x, y = self.center[i] + self.normal[i] * ypx
                mass = float(np.exp(rng.uniform(*np.log(spec.get("mass", [0.4, 3.0])))))
                items.append((x, y, size, mass, rng.uniform(0, np.pi / 2)))
        return items

    def make_shuttles(self, rng):
        """트랙을 가로질러 왕복하는 장애물. 반환 [(점 인덱스, 진폭px, 주기s, 위상, 반지름), ...]
        고정: "shuttles": [[트랙비율, 주기초, 위상 0~1, 반지름], ...]
        랜덤: "random_shuttles": {"count": 3, "period": [1.6, 3.0], "r": [10, 14]}"""
        items = []
        rows = [(f, T, ph, r) for f, T, ph, r in self.raw.get("shuttles", [])]
        spec = self.raw.get("random_shuttles")
        if spec:
            for f in np.sort(rng.uniform(spec.get("clear_start", 0.1), 0.95, spec["count"])):
                rows.append((f, rng.uniform(*spec.get("period", [1.6, 3.0])), rng.uniform(0, 1),
                             rng.uniform(*spec.get("r", [10, 14]))))
        for f, T, ph, r in rows:
            i = int(f * self.n) % self.n
            amp = max(0.0, self.half[i] - r - 3)          # 벽 바로 앞까지 쓸고 지나감
            items.append((i, amp, float(T), float(ph), float(r)))
        return items

    def nearest_idx(self, pos):
        return int(np.argmin(np.linalg.norm(self.center - np.asarray(pos), axis=1)))

    # ------------------------------------------------------------------
    def project(self, pos, hint_idx, back=6, ahead=20):
        """차 위치를 hint_idx 주변 중심선 구간에 투영.

        반환: (구간 인덱스, 중심선 위 거리 s, 중심선까지 거리)
        전체 트랙이 아니라 주변만 찾기 때문에, 트랙이 가까이 붙어 있는
        헤어핀에서도 옆 차선으로 '순간이동'하지 않는다.
        """
        idx = (hint_idx + np.arange(-back, ahead)) % self.n
        a = self.center[idx]
        v = self.seg_vec[idx]
        p = np.asarray(pos)
        t = np.clip(np.einsum("ij,ij->i", p - a, v) / (self.seg_len[idx] ** 2), 0, 1)
        proj = a + v * t[:, None]
        d = np.linalg.norm(p - proj, axis=1)
        k = int(np.argmin(d))
        i = int(idx[k])
        return i, float(self.s_start[i] + t[k] * self.seg_len[i]), float(d[k])

    def point_at(self, s):
        """중심선 위 거리 s 지점의 좌표 (한 바퀴 넘어가면 감아서)."""
        s = s % self.total_length
        i = int(np.searchsorted(self.s_start, s, side="right") - 1)
        t = (s - self.s_start[i]) / self.seg_len[i]
        return self.center[i] + self.seg_vec[i] * t


def load_track_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
