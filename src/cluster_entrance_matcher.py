# cluster_entrance_matcher.py
import numpy as np
import pandas as pd
import requests
import logging
import json
from typing import Optional

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class ClusterEntranceMatcher:
    """
    points_df (lat, lon, cluster) 와 entrances_df (lat, lon, 기타 컬럼) 를 받아
    각 group 별로 '그룹 내 어떤 지점' 과 '가장 가까운 entrance' 를 매칭해 반환한다.

    주요 메서드:
      - match(strategy='min_point', sample_rate=None, use_osrm=False, osrm_url=None)
        strategy: 'min_point' (기본) | 'centroid'
    """
    def __init__(self, points_df: pd.DataFrame, entrances_df: pd.DataFrame):
        # 입력 검증
        if not {'lat','lon','cluster'}.issubset(points_df.columns):
            raise ValueError("points_df는 'lat','lon','group' 컬럼을 포함해야 합니다.")
        if not {'lat','lon'}.issubset(entrances_df.columns):
            raise ValueError("entrances_df는 'lat','lon' 컬럼을 포함해야 합니다.")

        self.points_df = points_df.copy().reset_index(drop=True)
        self.entrances_df = entrances_df.copy().reset_index(drop=True)
        # np arrays for computation
        self._entrances_coords = self.entrances_df[['lat','lon']].values  # shape (M,2)

    # ---------- 헬퍼: Haversine (meters) ----------
    @staticmethod
    def _haversine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        a: (N,2) [lat,lon], b: (M,2) [lat,lon]
        returns: (N,M) distances in meters
        """
        if a.size == 0 or b.size == 0:
            return np.zeros((a.shape[0], b.shape[0]))
        # radians
        lat1 = np.radians(a[:,0])[:,None]  # (N,1)
        lon1 = np.radians(a[:,1])[:,None]
        lat2 = np.radians(b[:,0])[None,:]  # (1,M)
        lon2 = np.radians(b[:,1])[None,:]

        dlat = lat2 - lat1  # (N,M)
        dlon = lon2 - lon1
        R = 6371000.0  # Earth radius in meters

        sin_dlat = np.sin(dlat/2.0)
        sin_dlon = np.sin(dlon/2.0)
        a_hav = sin_dlat**2 + np.cos(lat1) * np.cos(lat2) * sin_dlon**2
        c = 2 * np.arctan2(np.sqrt(a_hav), np.sqrt(1 - a_hav))
        return R * c  # (N,M)

    # ---------- 헬퍼: OSRM single-table (sources list -> destinations list) ----------
    def _osrm_table(self, sources: np.ndarray, destinations: np.ndarray, osrm_url: str):
        """
        sources: (S,2)[lat,lon], destinations: (D,2)
        returns dict {'distances': np.array(S,D) meters, 'durations': np.array(S,D) seconds}
        """
        if osrm_url is None:
            raise ValueError("osrm_url must be provided for OSRM calls")

        coords_list = [f"{lon},{lat}" for lat,lon in np.vstack([sources, destinations])]
        S = len(sources)
        D = len(destinations)
        sources_idx = ";".join(str(i) for i in range(S))
        destinations_idx = ";".join(str(i) for i in range(S, S + D))
        url = f"{osrm_url}/table/v1/driving/{';'.join(coords_list)}?sources={sources_idx}&destinations={destinations_idx}&annotations=distance,duration"

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        distances = np.array(data.get('distances', []))  # shape (S,D)
        durations = np.array(data.get('durations', []))
        return {'distances': distances, 'durations': durations}

    # ---------- 공개 메서드 ----------
    def match(self, sample_rate: Optional[int], osrm_url: str, exclude_noise: bool) -> list[dict]:
        """
        각 클러스터별로 대표 방문지점과 가장 가까운 진입점 매칭 후 JSON 반환
        """
        groups = sorted(self.points_df['cluster'].unique())
        if exclude_noise and -1 in groups:
            groups = [g for g in groups if g != -1]

        clusters_df = []

        for g in groups:
            group_df = self.points_df[self.points_df['cluster'] == g]
            if group_df.shape[0] == 0:
                continue

            pts = group_df[['lat','lon']].values  # (N,2)

            # 샘플링
            if sample_rate is not None and sample_rate > 0 and len(pts) > sample_rate:
                idxs = np.linspace(0, len(pts)-1, sample_rate, dtype=int)
                cand_pts = pts[idxs]
            else:
                cand_pts = pts

            # Haversine 거리 기준 상위 N개 entrance 선택
            N = 6
            dmat = self._haversine_matrix(cand_pts, self._entrances_coords)  # (P, M)
            flat_indices = np.argsort(dmat, axis=None)[:N]  # 가장 가까운 N개 flat index
            p_indices, e_indices = np.unravel_index(flat_indices, dmat.shape)  # 각 인덱스 쌍

            entrances_info = []

            for p_idx, e_idx in zip(p_indices, e_indices):
                rep_lat, rep_lon = float(cand_pts[p_idx,0]), float(cand_pts[p_idx,1])
                entr_lat, entr_lon = float(self._entrances_coords[e_idx,0]), float(self._entrances_coords[e_idx,1])
                dist_m = float(dmat[p_idx, e_idx])
                duration_s = np.nan

                # OSRM Table API 호출
                tbl = self._osrm_table(np.array([[rep_lat, rep_lon]]),
                                    np.array([[entr_lat, entr_lon]]),
                                    osrm_url)
                if tbl['durations'].size > 0:
                    duration_s = float(tbl['durations'][0,0])
                if tbl['distances'].size > 0:
                    dist_m = float(tbl['distances'][0,0])

                # entrance 메타데이터 안전하게 처리
                entr_meta_row = self.entrances_df.reset_index().iloc[e_idx].to_dict()
                safe_entr_meta = {k: (None if pd.isna(v) else v) for k, v in entr_meta_row.items()}

                entrances_info.append({
                    "lat": entr_lat,
                    "lon": entr_lon,
                    "distance_m": dist_m,
                    "duration_s": duration_s,
                    "meta": safe_entr_meta
                })

            # lat, lon 중복 제거
            seen_coords = set()
            unique_entrances = []
            for ent in entrances_info:
                coord = (ent["lat"], ent["lon"])
                if coord not in seen_coords:
                    seen_coords.add(coord)
                    unique_entrances.append(ent)

            # duration_s 기준으로 오름차순 정렬
            unique_entrances.sort(key=lambda x: x["duration_s"] if not np.isnan(x["duration_s"]) else np.inf)

            # point 메타데이터 안전 처리 후 JSON 구조 생성
            cluster_dict = {
                "cluster_id": int(g),
                "representative": {"lat": rep_lat, "lon": rep_lon},
                "entrances": unique_entrances,
                "points": [
                    {
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"]),
                        "meta": {
                            k: (None if pd.isna(v) else v)
                            for k, v in row.to_dict().items()
                            if k != "group"
                        }
                    } for _, row in group_df.iterrows()
                ]
            }

            clusters_df.append(cluster_dict)

        return clusters_df