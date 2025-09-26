import numpy as np
import requests
import logging

class IsolatedFilter:
    def __init__(self, segments, osrm_url, max_travel_time, isolation_distance, batch_size=10):
        self.segments = segments
        self.final_points = None
        self.final_entrances = None
        self.removed_points = None  # 제거된 점 기록
        self.osrm_url = osrm_url
        self.max_travel_time = max_travel_time
        self.isolation_distance = isolation_distance
        self.batch_size = batch_size  # OSRM 호출 시 batch 단위

    def _get_travel_times(self, points, ref_points):
        """
        points: (P,2) 배열
        ref_points: (R,2) 배열
        """
        if len(points) == 0 or len(ref_points) == 0:
            return np.zeros((len(points), len(ref_points)))

        all_durations = []

        # 배치 단위로 OSRM 호출
        for i in range(0, len(points), self.batch_size):
            batch_points = points[i:i+self.batch_size]
            coords_a = [f"{lon},{lat}" for lat, lon in batch_points]
            coords_b = [f"{lon},{lat}" for lat, lon in ref_points]

            all_coords = coords_a + coords_b
            sources = ";".join(str(j) for j in range(len(coords_a)))
            destinations = ";".join(str(j) for j in range(len(coords_a), len(coords_a) + len(coords_b)))

            url = f"{self.osrm_url}/table/v1/driving/{';'.join(all_coords)}?sources={sources}&destinations={destinations}&annotations=duration"

            try:
                resp = requests.get(url)
                resp.raise_for_status()
                data = resp.json()
                durations = np.array(data['durations'])
                all_durations.append(durations)
            except Exception as e:
                logging.error(f"OSRM Table API 요청 실패: {e}")
                all_durations.append(np.full((len(batch_points), len(ref_points)), np.inf))

        if all_durations:
            return np.vstack(all_durations)
        return np.zeros((len(points), len(ref_points)))

    def filter(self):
        retained_points = []
        removed_points_list = []

        for seg in self.segments:
            points = seg.points
            entrances = getattr(seg, 'nearby_entrances', np.empty((0,2)))

            if len(points) == 0:
                continue

            # 1. 주변 방문지점 존재 여부 확인 (고립된 점과 이웃이 있는 점 분리)
            mask_near_points = np.zeros(len(points), dtype=bool)
            for i, p in enumerate(points):
                dists = np.linalg.norm(points - p, axis=1)
                if np.sum(dists < self.isolation_distance) > 1:
                    mask_near_points[i] = True
            
            # '이웃이 있는 점들'은 최종 보존 후보
            points_final = points[mask_near_points]
            
            # 📌 수정된 로직 시작: '이웃이 없는 고립된 점들'을 2단계 필터링 대상으로 설정
            points_to_check = points[~mask_near_points] 
            
            
            # 2. OSRM로 최종 평가: '고립된 점들' 중에서 max_travel_time 기준으로 제거 대상을 한 번 더 선별
            if points_to_check.size > 0 and entrances.shape[0] > 0:
                
                # 고립된 점들과 진입점 사이의 이동 시간 계산
                travel_times = self._get_travel_times(points_to_check, entrances)
                min_times = travel_times.min(axis=1)
                
                # max_travel_time을 초과하는 점들을 찾음 (즉, 접근성이 나쁜 고립점)
                mask_time = min_times > self.max_travel_time  # 조건을 반전하여 '제거할' 점을 찾음

                # '접근성이 나쁜 고립점'만 제거 목록에 추가
                removed_points_list.append(points_to_check[mask_time])
                
                # '접근성이 좋은 고립점'은 최종 보존 후보에 추가
                points_final = np.vstack([points_final, points_to_check[~mask_time]]) if points_final.size > 0 else points_to_check[~mask_time]

            # 3. 최종 보존 점 목록 추가
            if len(points_final) > 0:
                retained_points.append(points_final)

        self.final_points = np.vstack(retained_points) if retained_points else np.empty((0,2))
        self.final_entrances = np.vstack(
            [getattr(seg, 'nearby_entrances', np.empty((0,2))) for seg in self.segments if getattr(seg, 'nearby_entrances', np.empty((0,2))).size > 0]
        ) if self.segments else np.empty((0,2))

        # 제거된 점 합치기
        non_empty_removed = [pts for pts in removed_points_list if pts.size > 0]
        self.removed_points = np.vstack(non_empty_removed) if non_empty_removed else np.empty((0,2))

        logging.info(f"최종 방문 지점: {len(self.final_points)}개, 인접 진입점: {len(self.final_entrances)}개")
        logging.info(f"제거된 방문지점: {len(self.removed_points)}개")

        return self.final_points, self.final_entrances, self.removed_points