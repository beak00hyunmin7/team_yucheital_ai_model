"""
OpenFOAM 케이스(hillTerrainCase 등)의 실제 결과에서 (등고선, 배수) 이미지 쌍을 뽑아
ai_model/data/contours, ai_model/data/flow에 저장한다.

사용법 (ai_model/ 디렉터리에서):
    # 사용 가능한 시간 목록 확인
    python scripts/render_from_openfoam.py --case_dir ../../hillTerrainCase --list_times

    # 특정 시간의 (등고선, 배수) 이미지 쌍 저장
    python scripts/render_from_openfoam.py --case_dir ../../hillTerrainCase --time 25 --name hillTerrainCase_t25

    # 전체 시간 스텝을 프레임으로 뽑아 mp4로 미리보기 (실측 데이터 기반, AI 없이)
    python scripts/render_from_openfoam.py --case_dir ../../hillTerrainCase --video outputs/hillTerrainCase_flow.mp4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.openfoam_reader import parse_terrain_grid, parse_depth_grid, list_available_times
from src.utils.render_fields import render_contour_rgb, render_depth_rgb


def save_pair(case_dir, time, name, contours_dir, flow_dir, vmax=None):
    X, Y, Z = parse_terrain_grid(case_dir)
    depth = parse_depth_grid(case_dir, time)

    contour_rgb = render_contour_rgb(X, Y, Z)
    depth_rgb = render_depth_rgb(depth, vmax=vmax)

    os.makedirs(contours_dir, exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)
    from PIL import Image
    Image.fromarray(contour_rgb).save(os.path.join(contours_dir, f"{name}.png"))
    Image.fromarray(depth_rgb).save(os.path.join(flow_dir, f"{name}.png"))

    print(f"저장 완료: {contours_dir}/{name}.png, {flow_dir}/{name}.png "
          f"(depth max={depth.max():.4f}m, mean={depth.mean():.5f}m)")


def make_video(case_dir, out_path, fps=10):
    from src.utils.video_utils import frames_to_video

    times = list_available_times(case_dir)
    if not times:
        raise RuntimeError(f"{case_dir}에 alpha.water가 있는 시간 디렉토리가 없습니다.")

    depths = [parse_depth_grid(case_dir, t) for t in times]
    vmax = max(float(d.max()) for d in depths)

    frames = [render_depth_rgb(d, vmax=vmax) for d in depths]
    frames_to_video(frames, out_path, fps=fps)
    print(f"영상 저장 완료: {out_path} ({len(frames)}프레임, vmax={vmax:.4f}m)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", type=str, required=True)
    parser.add_argument("--time", type=str, default=None, help="예: 25, 12.5, 'latest'")
    parser.add_argument("--name", type=str, default=None, help="출력 파일명(확장자 제외)")
    parser.add_argument("--contours_dir", type=str, default="data/contours")
    parser.add_argument("--flow_dir", type=str, default="data/flow")
    parser.add_argument("--list_times", action="store_true")
    parser.add_argument("--video", type=str, default=None, help="지정하면 전체 시간 프레임을 이 경로에 mp4로 저장")
    args = parser.parse_args()

    if args.list_times:
        times = list_available_times(args.case_dir)
        print(f"사용 가능한 시간 ({len(times)}개):", times)
    elif args.video:
        make_video(args.case_dir, args.video)
    else:
        time = args.time
        if time is None or time == "latest":
            times = list_available_times(args.case_dir)
            if not times:
                raise RuntimeError("사용 가능한 시간 디렉토리가 없습니다.")
            time = times[-1]
        name = args.name or f"{os.path.basename(os.path.normpath(args.case_dir))}_t{time}"
        save_pair(args.case_dir, time, name, args.contours_dir, args.flow_dir)
