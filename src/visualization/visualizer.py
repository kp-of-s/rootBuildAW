import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
import matplotlib.cm as cm
import numpy as np

class Visualizer:
    def __init__(self, margin_ratio=0.2, figsize=(10,10)):
        self.margin_ratio = margin_ratio
        self.figsize = figsize

    def _create_gdf(self, coords):
        if coords is None or len(coords) == 0:
            return gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")
        df = gpd.GeoDataFrame(geometry=gpd.points_from_xy(coords[:,1], coords[:,0]), crs="EPSG:4326")
        return df

    def _get_colors(self, segments):
        """Segment group_id 기반 색상 매핑"""
        unique_ids = sorted(set(seg.group_id for seg in segments))
        cmap = cm.get_cmap('tab20', len(unique_ids))
        color_dict = {gid: cmap(i) for i, gid in enumerate(unique_ids)}
        return color_dict

    def plot_segments(self, segments, save_path="segments_map.png"):
        color_dict = self._get_colors(segments)
        fig, ax = plt.subplots(figsize=self.figsize)
        for seg in segments:
            gdf_seg = self._create_gdf(seg.points)
            if not gdf_seg.empty:
                gdf_seg.to_crs(3857).plot(ax=ax, color=color_dict[seg.group_id], markersize=30, alpha=0.6)
            gdf_hull = self._create_gdf(seg.hull)
            if not gdf_hull.empty:
                gdf_hull.to_crs(3857).plot(ax=ax, color=color_dict[seg.group_id], markersize=50, marker='x')
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12, crs="EPSG:3857")
        plt.title("Segments and Hulls")
        plt.savefig(save_path, dpi=200)
        plt.show()
        print(f"Segment 지도 저장 완료: {save_path}")

    def plot_nearby_entrances(self, segments, entrances_coords=None, save_path="nearby_map.png"):
        fig, ax = plt.subplots(figsize=self.figsize)

        # Segment 좌표 범위 계산
        all_seg_points = np.vstack([seg.points for seg in segments if seg.points.size > 0])
        if all_seg_points.size > 0 and entrances_coords is not None:
            lat_min, lat_max = all_seg_points[:,0].min(), all_seg_points[:,0].max()
            lon_min, lon_max = all_seg_points[:,1].min(), all_seg_points[:,1].max()
            # Segment 범위 안에 있는 진입점만 필터
            in_bounds = entrances_coords[
                (entrances_coords[:,0] >= lat_min) & (entrances_coords[:,0] <= lat_max) &
                (entrances_coords[:,1] >= lon_min) & (entrances_coords[:,1] <= lon_max)
            ]
            gdf_in_bounds = self._create_gdf(in_bounds)
            gdf_in_bounds.to_crs(3857).plot(ax=ax, color='red', markersize=50, alpha=0.3, marker='o')

        for seg in segments:
            # Segment 점
            gdf_seg = self._create_gdf(seg.points)
            if not gdf_seg.empty:
                gdf_seg.to_crs(3857).plot(ax=ax, color='blue', markersize=30, alpha=0.5)

            # Nearby Entrances (인접 성공)
            if hasattr(seg, 'nearby_entrances') and seg.nearby_entrances.size > 0:
                gdf_ne = self._create_gdf(seg.nearby_entrances)
                gdf_ne.to_crs(3857).plot(ax=ax, color='magenta', markersize=100, marker='*', edgecolor='black')

        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12, crs="EPSG:3857")
        plt.title("Segments with Nearby Entrances (Segment bounds gray)")
        plt.savefig(save_path, dpi=200)
        plt.show()
        print(f"NearbyEntrances 지도 저장 완료: {save_path}")


    def plot_final_points(self, final_points, final_entrances, removed_points=None, save_path="final_map.png"):
        fig, ax = plt.subplots(figsize=self.figsize)

        # 최종 방문지점
        gdf_points = self._create_gdf(final_points)
        if not gdf_points.empty:
            gdf_points.to_crs(3857).plot(ax=ax, color='green', markersize=50, alpha=0.8, label="Final Points")

        # 최종 인접 진입점
        gdf_entrances = self._create_gdf(final_entrances)
        if not gdf_entrances.empty:
            gdf_entrances.to_crs(3857).plot(ax=ax, color='purple', markersize=100, marker='*', label="Final Entrances")

        # 삭제된 방문지점 강조 표시
        if removed_points is not None and removed_points.size > 0:
            gdf_removed = self._create_gdf(removed_points)
            gdf_removed.to_crs(3857).plot(ax=ax, color='red', markersize=120, marker='X', edgecolor='black', alpha=0.9, label="Removed Points")

        # Basemap 추가
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=14, crs="EPSG:3857")

        plt.title("Final Visitation Map (Removed Points Highlighted)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.savefig(save_path, dpi=200)
        plt.show()
        print(f"최종 지도 저장 완료: {save_path}")

