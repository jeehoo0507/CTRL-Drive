"""로컬 평가. 대회 서버와 '똑같은' 채점 함수(racegame.scoring.evaluate)를 쓴다.
서버와 다른 건 트랙(연습 트랙 vs 비공개 트랙)과 시드뿐.

    python evaluate.py                                   # ./agent.py
    python evaluate.py --agent examples/rule_based       # 다른 에이전트
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import racegame  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default=HERE)
    p.add_argument("--tracks", nargs="+", default=racegame.list_tracks())
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--json", help="결과를 json으로 저장할 경로")
    a = p.parse_args()

    agent = racegame.load_agent(a.agent)
    results = racegame.evaluate(agent, a.tracks, seeds=a.seeds)
    print(racegame.format_table(results))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
