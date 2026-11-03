"""주행 리플레이를 mp4로 저장. 점수만 봐서는 왜 망했는지 모르니까 꼭 눈으로 확인!

    python watch.py                                  # 연습 트랙 전부 이어서 → replay.mp4
    python watch.py --tracks hairpin --out hp.mp4
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import racegame  # noqa: E402

FPS = int(round(1 / racegame.DT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default=HERE)
    p.add_argument("--tracks", nargs="+", default=racegame.list_tracks())
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--title", default=None, help="화면 왼쪽 위에 표시할 이름")
    p.add_argument("--out", default="replay.mp4")
    a = p.parse_args()

    import imageio.v2 as imageio
    agent = racegame.load_agent(a.agent)
    title = a.title or "AI RACE"
    writer = imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=None,
                                ffmpeg_params=["-preset", "veryfast", "-crf", "23"])
    frames = []
    for track in a.tracks:
        frames.clear()
        info = racegame.run_episode(agent, track, a.seed,
                                    on_frame=lambda env: frames.append(np.asarray(
                                        env.render(title))))
        for f in frames + [frames[-1]] * FPS:      # 마지막 장면 1초 정지
            writer.append_data(f)
        print(f"{info['track']:<10} score {racegame.track_score(info):6.2f} | "
              f"{'완주' if info['finished'] else '실패'} {info['time']}s")
    writer.close()
    print(f"저장 → {a.out}")


if __name__ == "__main__":
    main()
