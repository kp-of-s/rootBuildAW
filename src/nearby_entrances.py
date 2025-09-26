# nearby_entrances.py
import numpy as np
import requests
import logging
import os, json
from geopy.distance import geodesic
import glob

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def numpy_json_encoder(obj):
    # NumPy 정수 (np.int32, np.int64 등)
    if isinstance(obj, np.integer):
        return int(obj)
    # NumPy 부동 소수점 (np.float32, np.float64 등)
    elif isinstance(obj, np.floating):
        return float(obj)
    # NumPy 부울 (np.bool_)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # NumPy 배열 (np.ndarray)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # 처리할 수 없는 객체가 남아 있으면 TypeError를 발생시킴
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

class NearbyEntrances:
    """
    Segment를 받아서 인접한 진입점을 계산
    """
    def __init__(self, segment, entrances_coords, distance_threshold, time_threshold, internal_sample_rate, osrm_url):
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
        self.internal_sample_rate = internal_sample_rate
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

    def _filter_by_geodesic_distance(self, coords_to_check, all_entrances_coords):
        """
        1차: 세그먼트의 특징점(coords_to_check)을 기준으로 모든 진입점(all_entrances_coords) 중
            유클리드 거리 임계값(self.geodesic_threshold_m) 이내인 진입점만 필터링합니다.
        """
        
        # 클래스 속성에서 유클리드 거리 임계값(예: 5000m)을 가져옵니다.
        geodesic_threshold = getattr(self, 'geodesic_threshold_m', 5000)
        
        # 필터링 마스크 (all_entrances_coords와 같은 길이)
        geodesic_mask = np.full(len(all_entrances_coords), False, dtype=bool)

        # 모든 진입점을 순회하며 가장 가까운 check_point와의 거리를 계산
        for idx, ent_coord in enumerate(all_entrances_coords):
            min_geodesic_dist = float('inf')
            
            # O(N*M) 복잡도로, 실제 환경에서는 KDTree 등으로 최적화가 필요합니다.
            for check_coord in coords_to_check:
                # check_coord와 ent_coord는 (lat, lon) 형태라고 가정
                # geodesic 함수를 사용하여 미터(m) 단위의 거리를 계산
                dist = geodesic(check_coord, ent_coord).meters
                min_geodesic_dist = min(min_geodesic_dist, dist)
                
            if min_geodesic_dist <= geodesic_threshold:
                geodesic_mask[idx] = True
                
        filtered_entrances = all_entrances_coords[geodesic_mask]

        logging.info(f"Segment {self.segment.group_id}: 유클리드 필터링 후 {len(filtered_entrances)}개 진입점 OSRM 분석 대상으로 선정됨 (기준: {geodesic_threshold}m)")

        return filtered_entrances

    def compute_nearby(self):
        hull_points = self.segment.hull
        if len(hull_points) == 0 or len(self.entrances_coords) == 0:
            self.nearby_entrances = np.array([])
            logging.info(f"Segment {self.segment.group_id}: hull 또는 entrances_coords 없음")
            return self.nearby_entrances

        # 내부 점 샘플링 및 OSRM 출발지 목록 준비
        internal_points = self.segment.points[::max(1, len(self.segment.points)//self.internal_sample_rate)]
        coords_to_check = np.vstack([hull_points, internal_points])
        
        # 📌 1단계: 유클리드 거리 기반 1차 필터링
        filtered_entrances_coords = self._filter_by_geodesic_distance(coords_to_check, self.entrances_coords)
        
        if len(filtered_entrances_coords) == 0:
            self.nearby_entrances = np.array([])
            return self.nearby_entrances

        # 📌 2단계: OSRM Table API를 통한 운전 거리/시간 2차 검증 및 데이터 추적
        
        # 📌 2단계: OSRM Table API를 통한 운전 거리/시간 2차 검증 (배치별로 모든 결과 저장)
        # OSRM 요청은 1차 필터링된 목록(filtered_entrances_coords)만을 대상으로 진행

        # 모든 OSRM 요청의 결과를 저장할 리스트 (JSON 구성용)
        osrm_results_by_entrance = {tuple(c): [] for c in filtered_entrances_coords}
        coords_to_check_list = coords_to_check.tolist()

        nearby_coords = [] # 최종 인접 진입점 좌표를 저장

        batch_size = 10  # coords_to_check (출발지)의 배치 사이즈
        for i in range(0, len(coords_to_check), batch_size):
            batch_coords = coords_to_check[i:i+batch_size]
            coords_a = [f"{lon},{lat}" for lat, lon in batch_coords]
            coords_b = [f"{lon},{lat}" for lat, lon in filtered_entrances_coords] 
            
            # OSRM URL 구성 (기존과 동일)
            all_coords = coords_a + coords_b
            sources = ";".join(str(j) for j in range(len(coords_a)))
            destinations = ";".join(str(j) for j in range(len(coords_a), len(coords_a)+len(coords_b)))

            url = f"{self.osrm_url}/table/v1/driving/{';'.join(all_coords)}?sources={sources}&destinations={destinations}&annotations=distance,duration"

            try:
                resp = requests.get(url)
                resp.raise_for_status()
                data = resp.json()
                distances = np.array(data['distances']) # (배치 출발지) x (필터링된 진입점)
                durations = np.array(data['durations'])

                # 📌 JSON 구조 변경을 위해 모든 배치 결과를 저장
                for j, ent_coord_np in enumerate(filtered_entrances_coords):
                    ent_coord_tuple = tuple(ent_coord_np)
                    
                    # 현재 진입점까지의 거리/시간 행 (distances[:, j], durations[:, j])
                    
                    # check_points 상세 정보 추가
                    for k in range(len(batch_coords)):
                        dist = distances[k, j]
                        time = durations[k, j]
                        check_point_coord = coords_to_check_list[i + k]

                        osrm_results_by_entrance[ent_coord_tuple].append({
                            "check_point": check_point_coord,
                            "distance_m": dist,
                            "duration_s": time
                        })
                
                # 📌 최종 인접성 판단 (기존 로직 유지)
                min_distances = distances.min(axis=0)
                min_durations = durations.min(axis=0)
                mask = (min_distances <= self.distance_threshold) & (min_durations <= self.time_threshold)
                nearby_coords.append(filtered_entrances_coords[mask])


            except Exception as e:
                logging.error(f"Segment {self.segment.group_id} OSRM Table API 요청 실패: {e}")

        # 📌 3단계: 최종 결과 통합 및 JSON 구성
            
        # 최종 nearby_entrances 배열 생성 (기존과 동일)
        if nearby_coords:
            non_empty_coords = [arr for arr in nearby_coords if arr.size > 0]
            if non_empty_coords:
                all_nearby_coords = np.vstack(non_empty_coords)
                self.nearby_entrances = np.unique(all_nearby_coords, axis=0)
            else:
                self.nearby_entrances = np.array([])
        else:
            self.nearby_entrances = np.array([])

        # 📌 새로운 JSON 구조 생성
        json_entrances = []
        for ent_coord_np in filtered_entrances_coords:
            ent_coord_tuple = tuple(ent_coord_np)
            
            # 인접성 판단 여부 (최종 결과 배열에 있는지 확인)
            is_nearby = any(np.array_equal(ent_coord_np, nearby_coord) for nearby_coord in self.nearby_entrances)
            
            json_entrances.append({
                "entrance_coord": ent_coord_np.tolist(),
                "is_nearby": is_nearby,
                # OSRM 요청 결과를 모두 담은 check_points 리스트
                "check_points": osrm_results_by_entrance[ent_coord_tuple] 
            })

        # JSON 저장
        output = {
            "segment_id": int(self.segment.group_id),
            "entrances": json_entrances
        }
        """
        save_dir = "nearby_osrm"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"segment_{self.segment.group_id}_osrm.json")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=numpy_json_encoder)
        """
        logging.info(f"Segment {self.segment.group_id} 처리 완료: {len(self.nearby_entrances)} nearby entrances 저장됨 -> {save_path}")
        return self.nearby_entrances