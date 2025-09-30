import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
import pandas as pd
import matplotlib.cm as cm
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Point
import json

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


    def plot_nearby_entrances(self, segments, entrances_df, save_path="nearby_map.png"):
        current_crs = "EPSG:4326"

        # 1. 모든 진입점 정보를 읽고 GeoDataFrame (GDF) 생성 (루프 외부에서 한 번만)
        df_ent_info = pd.read_csv("data/IC.csv")
        
        # 좌표 정확도 문제 대비, 소수점 6자리 반올림
        df_ent_info["lat"] = df_ent_info["lat"].round(6)
        df_ent_info["lon"] = df_ent_info["lon"].round(6)

        geometry = [Point(xy) for xy in zip(df_ent_info['lon'], df_ent_info['lat'])]
        gdf_all_ent = gpd.GeoDataFrame(df_ent_info, geometry=geometry, crs=current_crs)
        
        # 인접 진입점을 구분하기 위한 키 미리 생성
        gdf_all_ent['coord_key'] = gdf_all_ent.apply(lambda row: (row['lat'], row['lon']), axis=1)

        fig, ax = plt.subplots(figsize=self.figsize)

        for seg in segments:
            # 2. Segment 점 시각화
            gdf_seg = self._create_gdf(seg.points)
            if gdf_seg.empty:
                continue
                
            # 3857로 변환하여 플롯 (ax의 좌표계가 3857로 설정됨)
            gdf_seg.to_crs(3857).plot(ax=ax, color='blue', markersize=30, alpha=0.5, label='Segment Points')

            
            # 3. 지도 범위 (Segment Bounding Box, WGS84) 계산 및 전체 진입점 필터링
            
            # seg.points가 (lat, lon)이므로, min/max는 (min_lat, min_lon), (max_lat, max_lon)
            min_lat, min_lon = seg.points.min(axis=0)
            max_lat, max_lon = seg.points.max(axis=0)
            
            # lon/lat 순서로 GDF 필터링 (gdf_all_ent.cx[min_lon:max_lon, min_lat:max_lat])
            gdf_in_extent = gdf_all_ent.cx[min_lon - 0.001:max_lon + 0.001, min_lat - 0.001:max_lat + 0.001].copy() # 약간의 여백 추가
            
            if gdf_in_extent.empty:
                continue
                
            # 4. '인접 진입점' 마커 표시를 위한 데이터 준비
            if hasattr(seg, 'nearby_entrances') and seg.nearby_entrances.size > 0:
                df_nearby = pd.DataFrame(seg.nearby_entrances, columns=["lat", "lon"]).round(6)
                df_nearby['coord_key'] = df_nearby.apply(lambda row: (row['lat'], row['lon']), axis=1)
                nearby_keys = set(df_nearby['coord_key'])
            else:
                nearby_keys = set()
                
            # 5. 지도 범위 내 진입점을 '선택됨(is_nearby)'과 '미선택됨'으로 구분
            gdf_in_extent['is_nearby'] = gdf_in_extent['coord_key'].apply(lambda k: k in nearby_keys)
            
            # 6. 시각화 (3857 변환 후 두 가지 색상으로 분리 플롯)
            
            # A. 미선택 진입점 (In-Boundary only) - 회색/원
            gdf_not_nearby_3857 = gdf_in_extent[~gdf_in_extent['is_nearby']].to_crs(3857)
            if not gdf_not_nearby_3857.empty:
                gdf_not_nearby_3857.plot(
                    ax=ax, color='gray', markersize=50, marker='o', alpha=0.6, label='In-Boundary Entrances'
                )

            # B. 선택된 인접 진입점 (Nearby Entrances) - 마젠타/별
            gdf_nearby_3857 = gdf_in_extent[gdf_in_extent['is_nearby']].to_crs(3857)
            if not gdf_nearby_3857.empty:
                gdf_nearby_3857.plot(
                    ax=ax, color='magenta', markersize=150, marker='*', edgecolor='black', label='Nearby (Selected) Entrances'
                )
                
            # 7. 이름/ID 표시 (모든 지도 범위 내 진입점에 대해 주석 처리)
            # 주석은 모든 gdf_in_extent_3857에 대해 적용
            gdf_all_ent_3857 = pd.concat([gdf_not_nearby_3857, gdf_nearby_3857])
            
            for _, row in gdf_all_ent_3857.iterrows():
                ax.annotate(
                    # '시설명' 컬럼 사용 (없으면 빈 문자열)
                    text=str(row.get("시설명", "")), 
                    xy=(row.geometry.x, row.geometry.y), 
                    xycoords=ax.transData,              
                    fontsize=10,                      # 글자 크기 10으로 조정
                    color="darkred",
                    fontfamily='Malgun Gothic'        # 한글 폰트 하드코딩
                )

        # 8. 배경지도 및 최종 설정
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12, crs="EPSG:3857")
        
        # Matplotlib의 상태 기반 인터페이스 사용
        plt.title("Segments with All In-Boundary and Selected Nearby Entrances")
        
        # 범례가 중복되지 않게 처리 (두 번 플롯했기 때문에)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')

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

    def plot_matches(self, clusters_df, entrances_df, save_name="matches_map"):

        entrances_safe = entrances_df.where(pd.notna(entrances_df), None).to_dict(orient='records')

        html_str = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Clusters Map</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
                // 데이터를 전역 변수로 설정
                const clustersData = {json.dumps(clusters_df)};
                const entrancesData = {json.dumps(entrances_safe)};
                const regionNameData = "{save_name}";
            </script>
            <script src="../clusters_map.js"></script>
            <script>
                // 페이지 로드 후 초기화 및 데이터 설정
                document.addEventListener('DOMContentLoaded', function() {{
                    
                    // 맵 초기화
                    const mapInitialized = initializeMap();

                    if (mapInitialized) {{
                        // 이벤트 설정
                        setupClickEvent();
                        setupDownloadButton();
                        // 데이터 설정
                        setData(clustersData, entrancesData, regionNameData);
                    }} else {{
                        console.error("Map initialization failed!");
                    }}
                }});
            </script>
        </head>
        <body>
            <div id="map" style="width:100%;height:800px;"></div>
            <button id="downloadBtn" style="position:absolute;top:10px;right:10px;z-index:1000;padding:10px;">
                Export clusters_json
            </button>
        </body>
        </html>
        """

        with open(f"./src/visualization/html/{save_name}.html", "w", encoding="utf-8-sig") as f:
            f.write(html_str)
        