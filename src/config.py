# config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """전체 시스템 설정을 관리하는 DTO 클래스"""
    
    # ===========================================
    # DBSCAN 클러스터링 설정 (Segment 클래스)
    # ===========================================
    eps: float = 0.01
    """DBSCAN 클러스터링 반경 (도 단위)
    - 사용처: Segment.create_segments_from_csv()
    - 높으면: 더 넓은 범위의 지점들이 하나의 클러스터로 묶임 (클러스터 수 감소)
    - 낮으면: 더 좁은 범위만 같은 클러스터로 인식 (클러스터 수 증가, 세분화)"""
    
    min_samples: int = 5
    """DBSCAN 최소 샘플 수
    - 사용처: Segment.create_segments_from_csv()
    - 높으면: 더 많은 지점이 모여야 클러스터로 인정 (노이즈 증가, 클러스터 수 감소)
    - 낮으면: 적은 지점도 클러스터로 인정 (노이즈 감소, 클러스터 수 증가)"""
    
    # ===========================================
    # NearbyEntrances 설정
    # ===========================================
    distance_threshold: float = 10000  # 미터
    """진입점과 클러스터 외곽점 간 최대 거리 (미터)
    - 사용처: NearbyEntrances.compute_nearby()
    - 높으면: 더 먼 진입점도 인접 진입점으로 인정 (진입점 수 증가)
    - 낮으면: 가까운 진입점만 인정 (진입점 수 감소, 더 엄격한 필터링)"""
    
    time_threshold: float = 1200  # 초
    """진입점과 클러스터 외곽점 간 최대 주행 시간 (초)
    - 사용처: NearbyEntrances.compute_nearby()
    - 높으면: 더 오래 걸리는 진입점도 인정 (진입점 수 증가)
    - 낮으면: 빠른 도달 가능한 진입점만 인정 (진입점 수 감소)"""
    
    internal_sample_rate: int = 10
    """클러스터 내부 점 샘플링 비율
    - 사용처: NearbyEntrances.compute_nearby()
    - 높으면: 더 적은 내부 점 사용 (계산 속도 향상, 정확도 감소)
    - 낮으면: 더 많은 내부 점 사용 (계산 속도 감소, 정확도 향상)"""
    
    # ===========================================
    # IsolatedFilter 설정
    # ===========================================
    max_travel_time: float = 1200  # 초 (20분)
    """진입점까지 최대 허용 주행 시간 (초)
    - 사용처: IsolatedFilter.filter()
    - 높으면: 더 먼 지점도 유지 (제거되는 지점 감소)
    - 낮으면: 가까운 지점만 유지 (제거되는 지점 증가, 더 엄격한 필터링)"""
    
    isolation_distance: float = 0.05  # 도 단위 (약 5.5km)
    """고립점 판단을 위한 최소 인접 거리 (도 단위)
    - 사용처: IsolatedFilter.filter()
    - 높으면: 더 먼 지점도 "인접"으로 인정 (고립점 감소)
    - 낮으면: 가까운 지점만 "인접"으로 인정 (고립점 증가)"""
    
    # ===========================================
    # OSRM 설정
    # ===========================================
    osrm_url: str = "http://localhost:5000"
    
    # ===========================================
    # ClusterEntranceMatcher 설정
    # ===========================================
    sample_rate: Optional[int] = 30
    """클러스터 내 대표점 샘플링 수
    - 사용처: ClusterEntranceMatcher.match()
    - 높으면: 더 많은 대표점 사용 (정확도 향상, 계산 시간 증가)
    - 낮으면: 적은 대표점 사용 (계산 시간 단축, 정확도 감소)
    - None: 모든 점 사용 (가장 정확하지만 느림)"""
    
    exclude_noise: bool = False
    """DBSCAN 노이즈 클러스터 제외 여부
    - 사용처: ClusterEntranceMatcher.match()
    - True: 노이즈로 분류된 지점들 제외 (깔끔한 결과)
    - False: 노이즈 지점들도 포함 (모든 지점 처리)"""

