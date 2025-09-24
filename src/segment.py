# segment.py
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class Segment:
    """
    DBSCAN으로 그룹핑된 임시 그룹 한 개를 나타내는 클래스.
    좌표와 외곽점 정보를 관리.
    """
    def __init__(self, points: np.ndarray, group_id: int):
        """
        Parameters
        ----------
        points : np.ndarray
            그룹에 속한 방문지점 좌표, shape=(N,2)
        group_id : int
            그룹 ID
        """
        self.group_id = group_id
        self.points = points
        self.hull = self._compute_hull(points)

    def _compute_hull(self, points: np.ndarray) -> np.ndarray:
        """
        ConvexHull을 이용해 외곽점 계산.
        점이 3개 미만이면 원본 좌표 그대로 반환.
        """
        if len(points) < 3:
            return points
        try:
            hull = ConvexHull(points)
            return points[hull.vertices]
        except Exception as e:
            logging.warning(f"ConvexHull 계산 실패: {e}")
            return points

    @staticmethod
    def create_segments_from_csv(points_df, eps: float = 0.0001, min_samples: int = 5):
        """
        전체 방문지점 CSV를 받아 DBSCAN으로 그룹핑하고
        Segment 인스턴스 리스트 반환.
        """
        coords = points_df[['lat','lon']].values

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        points_df['group'] = clustering.labels_

        segments = []
        for g in set(clustering.labels_):
            group_points = points_df[points_df['group'] == g][['lat','lon']].values
            segment = Segment(group_points, group_id=g)
            segments.append(segment)
            logging.info(f"Segment {g}: {len(group_points)} points, hull {len(segment.hull)} points")
        return segments