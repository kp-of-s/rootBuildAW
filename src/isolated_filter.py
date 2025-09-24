import numpy as np
import requests
import logging

class IsolatedFilter:
    def __init__(self, segments, osrm_url="http://localhost:5000", max_travel_time=1800):
        self.segments = segments
        self.final_points = None
        self.final_entrances = None
        self.removed_points = None  # ← 제거된 점 기록
        self.osrm_url = osrm_url
        self.max_travel_time = max_travel_time

    def _get_travel_times(self, points, ref_points):
        if len(points) == 0 or len(ref_points) == 0:
            return np.zeros((len(points), len(ref_points)))

        coords_a = [f"{lon},{lat}" for lat, lon in points]
        coords_b = [f"{lon},{lat}" for lat, lon in ref_points]

        all_coords = coords_a + coords_b
        sources = ";".join(str(i) for i in range(len(coords_a)))
        destinations = ";".join(str(i) for i in range(len(coords_a), len(coords_a) + len(coords_b)))

        url = f"{self.osrm_url}/table/v1/driving/{';'.join(all_coords)}?sources={sources}&destinations={destinations}&annotations=duration"

        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()
            durations = np.array(data['durations'])
            return durations
        except Exception as e:
            logging.error(f"OSRM Table API 요청 실패: {e}")
            return np.full((len(points), len(ref_points)), np.inf)

    def filter(self):
        retained_points = []
        removed_points_list = []

        for seg in self.segments:
            points = seg.points
            entrances = getattr(seg, 'nearby_entrances', np.empty((0,2)))

            if len(points) == 0:
                continue

            # 주변 방문지점 존재 여부 확인 (간단 거리 기준)
            mask_near_points = np.zeros(len(points), dtype=bool)
            for i, p in enumerate(points):
                dists = np.linalg.norm(points - p, axis=1)
                if np.sum(dists < 0.05) > 1:  # 자기 자신 제외
                    mask_near_points[i] = True
            points_filtered = points[mask_near_points]

            # 필터링 후 제거된 점
            removed_points_list.append(points[~mask_near_points])

            if len(points_filtered) == 0:
                continue

            points_final = points_filtered

            # OSRM로 최종 평가 (자동차 기준 30분 이상이면 제거)
            if points_final.size > 0 and entrances.shape[0] > 0:
                travel_times = self._get_travel_times(points_final, entrances)
                min_times = travel_times.min(axis=1)
                mask_time = min_times <= self.max_travel_time

                removed_points_list.append(points_final[~mask_time])  # OSRM 기준으로 제거된 점
                points_final = points_final[mask_time]

            if len(points_final) > 0:
                retained_points.append(points_final)

        self.final_points = np.vstack(retained_points) if retained_points else np.empty((0,2))
        self.final_entrances = np.vstack(
            [getattr(seg, 'nearby_entrances', np.empty((0,2))) for seg in self.segments if getattr(seg, 'nearby_entrances', np.empty((0,2))).size > 0]
        ) if self.segments else np.empty((0,2))

        # 제거된 점 합치기
        non_empty_removed = [pts for pts in removed_points_list if pts.size > 0]
        if non_empty_removed:
            self.removed_points = np.vstack(non_empty_removed)
        else:
            self.removed_points = np.empty((0, 2))

        logging.info(f"최종 방문 지점: {len(self.final_points)}개, 인접 진입점: {len(self.final_entrances)}개")
        logging.info(f"제거된 방문지점: {len(self.removed_points)}개")
        return self.final_points, self.final_entrances, self.removed_points
