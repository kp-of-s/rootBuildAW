OSRM 지도 적용

도커 설치 및 osm.pdf 다운

도로망 추출
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/south-korea-map.osm.pbf

MLD 전처리(다중 경유지 알고리즘)
Partition
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-partition /data/south-korea-map.osrm
Customize
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-customize /data/south-korea-map.osrm

도커 시작
docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/south-korea-map.osrm


# filter_outliers.py

## 개요
`filter_outliers.py`는 대규모 지점 데이터를 대상으로 **실제 도로망 접근성을 기반으로 외곽 지점을 필터링**하는 Python 모듈입니다.

- 전체 지점 중 **접근이 어려운 지역이나 외곽 지점**을 자동으로 제거
- 클러스터링과 Convex Hull을 활용하여 **지역 단위로 접근 가능성 판단**
- OSRM Table API를 활용하여 **실제 주행 거리 기준**으로 필터링

> 주 목적: 방문 지점이 넓게 흩어져 있을 때, **실제 접근 가능한 지점만 선택**하여 이후 경로 최적화(VRP/TSP)에 활용

---

## 주요 기능

1. **클러스터링 기반 지역 분리**
   - DBSCAN을 활용하여 지리적 밀집 지역을 자동 탐지
   - 각 클러스터 단위로 외곽 좌표를 추출

2. **클러스터 외곽 좌표 추출**
   - Convex Hull을 활용
   - 중앙을 가로지르는 도로/IC도 포함 가능

3. **진입로 후보 추출**
   - 각 클러스터 외곽 좌표와 모든 진입로 간 거리 계산
   - 개수 제한 없이 모든 후보 진입로를 고려

4. **OSRM 주행 거리 기반 필터링**
   - Table API를 활용하여 실제 도로망 기준 거리 계산
   - 최소 거리 ≤ threshold → 접근 가능
   - 접근 불가 클러스터 제거

5. **결과 저장**
   - 필터링된 지점 CSV 파일로 출력
   - 접근 가능한 지점만 포함

---

## 입력 파일

1. `points_csv` : 전체 방문 지점 CSV
   - 컬럼: `lat`, `lon`
2. `entrances_csv` : 진입로 CSV
   - 컬럼: `lat`, `lon`

---

## 출력 파일

- `output_csv` : 접근 가능한 지점만 포함된 CSV

---