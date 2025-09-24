# nearby_entrances.py
import numpy as np
import requests
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class NearbyEntrances:
    """
    Segment를 받아서 인접한 진입점을 계산
    """
    def __init__(self, segment, entrances_coords, distance_threshold=30000, time_threshold=6000, osrm_url="http://localhost:5000"):
        """
        Parameters
        ----------
        segment : Segment
            Segment 객체 (points + hull)
        entrances_coords : np.ndarray
            진입점 좌표 배열, shape=(M,2)
        distance_threshold : float
            최대 거리 기준 (m)
        time_threshold : float
            최대 주행 시간 기준 (초)
        osrm_url : str
            OSRM 서버 URL
        """
        self.segment = segment
        self.entrances_coords = entrances_coords
        self.distance_threshold = distance_threshold
        self.time_threshold = time_threshold
        self.osrm_url = osrm_url
        self.nearby_entrances = None

    def _snap_to_road(self, coords):
        """
        OSRM /nearest API를 이용해 좌표를 도로망에 snap
        coords : np.ndarray, shape=(N,2), (lat, lon)
        Returns np.ndarray, shape=(N,2)
        """
        if len(coords) == 0:
            return coords

        snapped = []
        for lat, lon in coords:
            url = f"{self.osrm_url}/nearest/v1/driving/{lon},{lat}?number=1"
            try:
                resp = requests.get(url)
                resp.raise_for_status()
                data = resp.json()
                snapped_coord = data['waypoints'][0]['location']  # [lon, lat]
                snapped.append([snapped_coord[1], snapped_coord[0]])  # lat, lon
            except Exception as e:
                logging.warning(f"OSRM /nearest snap 실패: {e}, 원래 좌표 사용")
                snapped.append([lat, lon])
        return np.array(snapped)

    def compute_nearby(self, internal_sample_rate: int = 10):
        """
        Segment 외곽점(hull) + 내부 점 일부를 사용하여 인접 진입점 계산
        일정 기준 이하인 진입점만 nearby_entrances에 저장
        Parameters
        ----------
        internal_sample_rate : int
            내부 점 샘플링 간격 (예: 10이면 내부 점 중 1/10만 사용)
        """
        hull_points = self.segment.hull
        if len(hull_points) == 0 or len(self.entrances_coords) == 0:
            self.nearby_entrances = np.array([])
            return self.nearby_entrances

        # 내부 점 샘플링
        internal_points = self.segment.points[::max(1, len(self.segment.points)//internal_sample_rate)]
        coords_to_check = np.vstack([hull_points, internal_points])

        # OSRM Table API 호출 준비
        coords_a = [f"{lon},{lat}" for lat, lon in coords_to_check]
        coords_b = [f"{lon},{lat}" for lat, lon in self.entrances_coords]

        all_coords = coords_a + coords_b
        sources = ";".join(str(i) for i in range(len(coords_a)))
        destinations = ";".join(str(i) for i in range(len(coords_a), len(coords_a) + len(coords_b)))

        url = f"{self.osrm_url}/table/v1/driving/{';'.join(all_coords)}?sources={sources}&destinations={destinations}&annotations=distance,duration"

        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()
            distances = np.array(data['distances'])
            durations = np.array(data['durations'])

            # 최소 거리/시간 기준 필터
            min_distances = distances.min(axis=0)
            min_durations = durations.min(axis=0)
            # print(min_distances)
            # print(min_durations)
            # print(self.distance_threshold)
            # print(self.time_threshold)
            mask = (min_distances <= self.distance_threshold) & (min_durations <= self.time_threshold)
            self.nearby_entrances = self.entrances_coords[mask]
            logging.info(f"Segment {self.segment.group_id}: {len(self.nearby_entrances)} nearby entrances found")
        except Exception as e:
            logging.error(f"OSRM Table API 요청 실패: {e}")
            self.nearby_entrances = np.array([])

        return self.nearby_entrances
