# cluster_entrance_matcher.py
import numpy as np
import pandas as pd
import requests
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class ClusterEntranceMatcher:
    """
    points_df (lat, lon, group) 와 entrances_df (lat, lon, 기타 컬럼) 를 받아
    각 group 별로 '그룹 내 어떤 지점' 과 '가장 가까운 entrance' 를 매칭해 반환한다.

    주요 메서드:
      - match(strategy='min_point', sample_rate=None, use_osrm=False, osrm_url=None)
        strategy: 'min_point' (기본) | 'centroid'
    """
    def __init__(self, points_df: pd.DataFrame, entrances_df: pd.DataFrame):
        # 입력 검증
        if not {'lat','lon','group'}.issubset(points_df.columns):
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
    def match(self,
              strategy: str = 'min_point',
              sample_rate: Optional[int] = None,
              use_osrm: bool = False,
              osrm_url: Optional[str] = None,
              exclude_noise: bool = True) -> pd.DataFrame:
        """
        cluster별 가장 가까운 entrance 선택

        Parameters
        ----------
        strategy : 'min_point'|'centroid'
            'min_point' : 그룹 내 (샘플링된) 모든 점을 후보로 하여 실제 최단 거리를 기준으로 entrance 선택 (권장)
            'centroid'  : 그룹 centroid(평균 좌표)만 사용하여 entrance 선택 (빠름)
        sample_rate : None or int
            min_point 전략에서 그룹 내 점을 샘플링할 때 사용.
            None이면 모든 점 사용. int이면 각 그룹에서 최대 sample_rate 개의 점을 균등 샘플링.
        use_osrm : bool
            선택된 (group point, entrance) 쌍에 대해 OSRM를 호출하여 duration(초) 반환 여부.
            (OSRM 호출은 선택된 쌍 개수만큼 수행)
        osrm_url : str or None
            OSRM 서버 URL (use_osrm=True일 때 필수)
        exclude_noise : bool
            group == -1 인 노이즈 그룹을 제외할지 여부
        Returns
        -------
        DataFrame with columns:
            group, rep_lat, rep_lon, entrance_idx, entr_lat, entr_lon, distance_m, duration_s (if use_osrm else NaN)
        """
        groups = sorted(self.points_df['group'].unique())
        if exclude_noise and -1 in groups:
            groups = [g for g in groups if g != -1]

        results = []
        for g in groups:
            group_df = self.points_df[self.points_df['group'] == g]
            if group_df.shape[0] == 0:
                continue

            pts = group_df[['lat','lon']].values  # (N,2)
            # 샘플링 처리 (균등 추출)
            if sample_rate is not None and sample_rate > 0 and len(pts) > sample_rate:
                idxs = np.linspace(0, len(pts)-1, sample_rate, dtype=int)
                cand_pts = pts[idxs]
            else:
                cand_pts = pts

            if strategy == 'centroid':
                rep = np.array([[pts[:,0].mean(), pts[:,1].mean()]])  # (1,2)
                # compute haversine from rep to all entrances:
                dists = self._haversine_matrix(rep, self._entrances_coords)  # (1,M)
                j = int(np.argmin(dists[0]))
                rep_lat, rep_lon = float(rep[0,0]), float(rep[0,1])
                entr_lat, entr_lon = float(self._entrances_coords[j,0]), float(self._entrances_coords[j,1])
                dist_m = float(dists[0,j])
                duration_s = np.nan
                # optional OSRM verification
                if use_osrm:
                    if osrm_url is None:
                        raise ValueError("osrm_url required when use_osrm=True")
                    tbl = self._osrm_table(rep, self._entrances_coords, osrm_url)
                    duration_s = float(tbl['durations'][0, j]) if tbl['durations'].size else np.nan
                    dist_m = float(tbl['distances'][0, j]) if tbl['distances'].size else dist_m

            elif strategy == 'min_point':
                # compute full (cand_pts x entrances) haversine matrix
                dmat = self._haversine_matrix(cand_pts, self._entrances_coords)  # (P, M)
                flat_idx = int(np.argmin(dmat))  # argmin over flattened array
                p_idx, e_idx = divmod(flat_idx, dmat.shape[1])
                rep_lat, rep_lon = float(cand_pts[p_idx,0]), float(cand_pts[p_idx,1])
                entr_lat, entr_lon = float(self._entrances_coords[e_idx,0]), float(self._entrances_coords[e_idx,1])
                dist_m = float(dmat[p_idx, e_idx])
                duration_s = np.nan
                # optional OSRM verification for the selected pair only
                if use_osrm:
                    if osrm_url is None:
                        raise ValueError("osrm_url required when use_osrm=True")
                    try:
                        tbl = self._osrm_table(np.array([[rep_lat, rep_lon]]), np.array([[entr_lat, entr_lon]]), osrm_url)
                        duration_s = float(tbl['durations'][0,0]) if tbl['durations'].size else np.nan
                        dist_m = float(tbl['distances'][0,0]) if tbl['distances'].size else dist_m
                    except Exception as e:
                        logging.warning(f"OSRM verification failed for group {g}: {e}")
                e_idx = int(e_idx)
            else:
                raise ValueError("strategy must be 'min_point' or 'centroid'")

            # 기록 (entrance index refers to index in entrances_df)
            results.append({
                'group': int(g),
                'rep_lat': rep_lat,
                'rep_lon': rep_lon,
                'entrance_idx': int(e_idx),
                'entr_lat': entr_lat,
                'entr_lon': entr_lon,
                'distance_m': dist_m,
                'duration_s': duration_s
            })

        df_res = pd.DataFrame(results, columns=['group','rep_lat','rep_lon','entrance_idx','entr_lat','entr_lon','distance_m','duration_s'])
        # join with entrances_df to include original entrance metadata (if needed)
        if not df_res.empty:
            # map entrance_idx to original entrances row
            entr_meta = self.entrances_df.reset_index().rename(columns={'index':'entrance_idx'})
            df_out = df_res.merge(entr_meta, on='entrance_idx', how='left', suffixes=('','_entr'))
        else:
            df_out = df_res
        return df_out