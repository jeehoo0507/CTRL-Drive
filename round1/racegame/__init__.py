"""racegame: 탑뷰 자동차 레이스 AI 대회용 환경.

    import racegame
    env = racegame.make("oval")
    obs = env.reset(seed=0)
    obs, reward, done, info = env.step([steer, throttle])   # 둘 다 -1 ~ 1
"""
from .env import RaceEnv, list_tracks, DT, MAX_STEPS
from .sensors import OBS_DIM                           # 12: 레이캐스트 9 + 속도 + 방향 2
BASE_OBS_DIM = OBS_DIM
from .scoring import evaluate, load_agent, format_table, track_score, run_episode

ACTION_DIM = 2  # [steer, throttle]


def make(track, **kwargs):
    return RaceEnv(track, **kwargs)


__all__ = ["make", "RaceEnv", "list_tracks", "evaluate", "load_agent", "format_table",
           "track_score", "run_episode", "OBS_DIM", "BASE_OBS_DIM", "ACTION_DIM", "DT", "MAX_STEPS"]
