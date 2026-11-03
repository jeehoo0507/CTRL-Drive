"""공식 채점. 참가자 로컬과 대회 서버가 '이 파일 하나'를 같이 쓴다.
서버는 tracks / seeds 만 비공개 값으로 바꿔서 호출한다.
"""
import importlib.util
import os
import sys
import time
import uuid

import numpy as np

from .env import RaceEnv

FINISH_BASE = 100.0
TIME_BONUS_FROM = 60.0     # 완주 시 (60 - 기록초) 만큼 보너스


def load_agent(path):
    """agent.py 파일(또는 그 파일이 있는 폴더)에서 Agent 클래스를 불러와 생성."""
    if os.path.isdir(path):
        path = os.path.join(path, "agent.py")
    path = os.path.abspath(path)
    folder = os.path.dirname(path)
    spec = importlib.util.spec_from_file_location(f"agent_{uuid.uuid4().hex[:8]}", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, folder)
    try:
        spec.loader.exec_module(module)
        return module.Agent()
    finally:
        sys.path.remove(folder)


def track_score(info):
    if info["finished"]:
        t = info["time"] - info.get("coin_time", 0.0)          # 전략 트랙: 코인만큼 기록 단축
        return FINISH_BASE + max(0.0, TIME_BONUS_FROM * info.get("laps", 1) - t)
    return 100.0 * info["progress"]


def run_episode(agent, track, seed, on_frame=None):
    env = RaceEnv(track)
    obs = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()
    if on_frame:
        on_frame(env)
    done, info, act_time = False, {}, 0.0
    while not done:
        t0 = time.perf_counter()
        action = agent.act(obs)
        act_time = max(act_time, time.perf_counter() - t0)
        obs, _, done, info = env.step(action)
        if on_frame:
            on_frame(env)
    info = dict(info, max_act_ms=round(act_time * 1000, 2))
    return info


def evaluate(agent, tracks, seeds=(0, 1, 2)):
    """반환: {트랙: {"score", "runs": [info...]}, ..., "total": 평균점수}"""
    results = {}
    for track in tracks:
        runs = []
        for seed in seeds:
            info = run_episode(agent, track, seed)
            info["score"] = round(track_score(info), 2)
            runs.append(info)
        name = runs[0]["track"]
        results[name] = {"score": round(float(np.mean([r["score"] for r in runs])), 2),
                         "category": runs[0]["category"], "runs": runs}
    results["total"] = round(float(np.mean([v["score"] for v in results.values()])), 2)
    return results


def format_table(results):
    lines = [f"{'track':<12}{'score':>8}  {'finish':>6}  {'best time':>9}  {'avg progress':>12}",
             "-" * 54]
    for name, r in results.items():
        if name == "total":
            continue
        runs = r["runs"]
        fin = sum(x["finished"] for x in runs)
        times = [x["time"] for x in runs if x["finished"]]
        best = f"{min(times):.2f}s" if times else "-"
        prog = np.mean([x["progress"] for x in runs]) * 100
        lines.append(f"{name:<12}{r['score']:>8.2f}  {fin:>3}/{len(runs):<2}  {best:>9}  {prog:>11.1f}%")
    lines += ["-" * 54, f"{'TOTAL':<12}{results['total']:>8.2f}"]
    return "\n".join(lines)
