# app.py
import numpy as np
import logging

from src.segment import Segment
from src.nearby_entrances import NearbyEntrances
from src.isolated_filter import IsolatedFilter
from src.visualization.visualizer import Visualizer
from src.cluster_entrance_matcher import ClusterEntranceMatcher
from src.config import Config
import json
import pandas as pd
import glob
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def main(
        points_df,           # 전체 방문 지점 CSV 데이터프레임
        entrances_df,        # 진입점 CSV 데이터프레임
        config: Config,      # 설정 객체
        region_name
    ):
    viz = Visualizer()

    # Segment 생성
    segments = Segment.create_segments_from_csv(points_df, eps=config.eps, min_samples=config.min_samples)
    # viz.plot_segments(segments)  # Segment + Hull 시각화

    # 진입점 좌표 로드
    entrances_coords = entrances_df[['lat', 'lon']].values

    # NearbyEntrances 계산
    for seg in segments:
        ne = NearbyEntrances(
            seg,
            entrances_coords,
            distance_threshold=config.distance_threshold,
            time_threshold=config.time_threshold,
            internal_sample_rate=config.internal_sample_rate,
            osrm_url=config.osrm_url
        )
        seg.nearby_entrances = ne.compute_nearby()
    # viz.plot_nearby_entrances(segments, entrances_df)  # Segment + NearbyEntrances 시각화


    # 4️⃣ 고립점 필터링
    filterer = IsolatedFilter(
        segments,
        osrm_url=config.osrm_url,
        max_travel_time=config.max_travel_time,
        isolation_distance=config.isolation_distance
    )
    final_points, final_entrances, removed_points = filterer.filter()

    logging.info("필?터링")
    # viz.plot_final_points(final_points, final_entrances, removed_points)  # 최종 결과 시각화

    final_entrances_df = get_final_entrances_df(final_entrances, entrances_df)

    matcher = ClusterEntranceMatcher(points_df, final_entrances_df)

    clusters_json = matcher.match(
        sample_rate=config.sample_rate,
        osrm_url=config.osrm_url,
        exclude_noise=config.exclude_noise
    )

    # viz.plot_matches(clusters_json)

    with open(f"{region_name}.json", "w", encoding="utf-8") as f:
        json.dump(clusters_json, f, ensure_ascii=False, indent=2)
    

    return final_points, final_entrances

def get_final_entrances_df(final_entrances: np.ndarray, entrances_df: pd.DataFrame) -> pd.DataFrame:
    if final_entrances.size == 0:
        return pd.DataFrame(columns=entrances_df.columns)

    # np.isclose를 사용해 위도/경도 비교 (소수점 오차 방지)
    mask = np.zeros(len(entrances_df), dtype=bool)
    for lat, lon in final_entrances:
        mask |= np.isclose(entrances_df['lat'].values, lat) & np.isclose(entrances_df['lon'].values, lon)

    return entrances_df[mask].copy()


# -----------------------------
# 실행 예시
# -----------------------------
if __name__ == "__main__":
    import os
    import pandas as pd

    # 1️⃣ 지역 목록 읽기
    # region_list_df = pd.read_csv("data/region.csv")
    # regions = region_list_df['region'].tolist()

    csv_files = glob.glob("data/region/**/*.csv", recursive=True)
    regions = [Path(f).stem for f in csv_files]

    # 2️⃣ 콘솔에서 선택
    print("지역 목록:")
    for i, r in enumerate(regions):
        print(f"{i}: {r}")
    
    selected_idx = None
    while selected_idx is None:
        try:
            choice = int(input("조회할 지역 번호를 입력하세요: "))
            if 0 <= choice < len(regions):
                selected_idx = choice
            else:
                print("번호 범위를 벗어났습니다.")
        except ValueError:
            print("숫자를 입력하세요.")

    region_name = regions[selected_idx]
    region_dir = os.path.join("data", "region", region_name)

    # 해당 폴더의 모든 CSV 파일 탐색
    csv_files = glob.glob(os.path.join(region_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"{region_dir} 폴더에 CSV 파일이 없습니다.")

    # 여러 개면 concat, 하나면 바로 읽음
    dfs = [pd.read_csv(f) for f in csv_files]
    points_df = pd.concat(dfs, ignore_index=True)

    # 기존 IC.csv는 동일
    entrances_csv = "data/IC.csv"
    entrances_df = pd.read_csv(entrances_csv)

    # 3️⃣ 설정 생성
    config = Config()

    
    # 4️⃣ main 실행
    final_points, final_entrances = main(
        points_df,
        entrances_df,
        config,
        region_name=region_name
    )

