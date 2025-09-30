let mapDiv = null;
let selectedEntrances = [];
let clusters_json = [];
let entrances_df = [];
let regionName = ""; // 지역 이름 (예: "서울특별시")

// DOM 요소 초기화 함수
function initializeMap() {
    mapDiv = document.getElementById('map');    
    if (!mapDiv) {
        console.error('Map div not found');
        return false;
    }
    
    // 초기 빈 지도
    Plotly.newPlot(mapDiv, [], {
        mapbox: { style: "carto-positron", center: {lat:37.5, lon:127}, zoom:12 },
        margin: {r:0,t:0,l:0,b:0}
    }).then(() => {
        setupClickEvent();
        setupDownloadButton();
    });

    return true;
}


// 데이터 설정 함수
function setData(clusters, entrances, region) {
    clusters_json = clusters;
    entrances_df = entrances;
    regionName = region;
    
    // 맵이 초기화되지 않았다면 초기화
    if (!mapDiv) {
        if (!initializeMap()) {
            console.error('Failed to initialize map');
            return;
        }
    }
    
    drawPoints();
}

// 포인트 그리기
function drawPoints() {
    if (!mapDiv) {
        console.error('Map div not initialized');
        return;
    }
    
    const points = [];

    clusters_json.forEach(cluster => {
        // points는 표시용만
        cluster.points?.forEach(pt => {
            points.push({
                lat: pt.lat,
                lon: pt.lon,
                cluster_id: cluster.cluster_id,
                name: pt.meta?.name || "",
                isEditable: false,
                inCluster: false
            });
        });

        // entrances는 표시 + 편집 가능
        cluster.entrances?.forEach(ent => {
            points.push({
                lat: ent.lat,
                lon: ent.lon,
                cluster_id: cluster.cluster_id,
                name: ent.meta?.시설명 || "",
                isEntrance: true,
                inCluster: true
            });
        });
    });

    // entrances_df 후보 진입점 추가
    entrances_df.forEach(ent => {
        const existsInCluster = clusters_json.some(cluster =>
            cluster.entrances?.some(cEnt => cEnt.lat === ent.lat && cEnt.lon === ent.lon)
        );
        if(!existsInCluster) {
            points.push({
                lat: ent.lat,
                lon: ent.lon,
                cluster_id: undefined,
                name: ent.시설명 || "",
                isEntrance: true,
                inCluster: false
            });
        }
    });

    const colors = points.map(p => {
        if(p.isEntrance && p.inCluster) return 'black';
        if(p.isEntrance && !p.inCluster) return 'red';
        return p.cluster_id !== undefined ? `hsl(${(p.cluster_id*137.5)%360},100%,50%)` : 'blue'; 
        // 일반 포인트는 눈에 잘 띄는 색상(HSL 해시)
    });

    const trace = {
        type: 'scattermapbox',
        lat: points.map(p => p.lat),
        lon: points.map(p => p.lon),
        mode: 'markers',
        marker: {
            size: points.map(p => p.isEntrance ? 14 : 10),
            color: colors
        },
        text: points.map(p => `${p.name || ""} (Cluster: ${p.cluster_id ?? "-"})`),
        hoverinfo: 'text',
        customdata: points.map(p => p)
    };

    Plotly.react(mapDiv, [trace], mapDiv.layout);
}

// 정보창 생성
function showEntranceInfo(entrance) {
    if (!entrance.isEntrance) return; // 편집 불가 포인트는 무시

    const oldDiv = document.getElementById('clusterSelectDiv');
    if(oldDiv) oldDiv.remove();

    const div = document.createElement('div');
    div.id = 'clusterSelectDiv';
    div.style.position = 'absolute';
    div.style.top = '50px';
    div.style.right = '20px';
    div.style.background = 'white';
    div.style.border = '1px solid black';
    div.style.padding = '10px';
    div.style.zIndex = 10000;

    const title = document.createElement('p');
    title.innerText = `진입점: ${entrance.name || ""}} s`;
    div.appendChild(title);

    // 클러스터 체크박스 생성
    clusters_json.forEach(cluster => {
        const label = document.createElement('label');
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.value = cluster.cluster_id;

        const exists = cluster.entrances?.some(ent => ent.lat === entrance.lat 
                                                        && ent.lon === entrance.lon);
        if (exists) cb.checked = true;

        label.appendChild(cb);
        label.appendChild(document.createTextNode(' Cluster ' + cluster.cluster_id));
        div.appendChild(label);
        div.appendChild(document.createElement('br'));
    });

    // 확인 버튼
    const confirmBtn = document.createElement('button');
    confirmBtn.innerText = '확인';
    // 기존 confirmBtn.onclick 내부
    confirmBtn.onclick = () => {
        const checked = Array.from(div.querySelectorAll('input[type=checkbox]:checked'))
                            .map(cb => parseInt(cb.value));

        clusters_json.forEach(cluster => {
            // 기존 제거
            cluster.entrances = cluster.entrances?.filter(ent => !(ent.lat === entrance.lat && ent.lon === entrance.lon)) || [];
            sourceEntrance = entrances_df.find(ent => ent.lat === entrance.lat && ent.lon === entrance.lon);
            if(checked.includes(cluster.cluster_id)) {
                const newEntrance = {
                    lat: sourceEntrance.lat,
                    lon: sourceEntrance.lon,
                    meta: {
                        시설명: sourceEntrance.시설명 || "",
                        노선명: sourceEntrance.노선명 || "",
                        유형: sourceEntrance.유형 || "",
                        노선방향: sourceEntrance.노선방향 ?? null,
                        lat: sourceEntrance.lat,
                        lon: sourceEntrance.lon
                    }
                };
                cluster.entrances.push(newEntrance);
            }
        });

        // selectedEntrances 업데이트
        selectedEntrances = [];
        clusters_json.forEach(cluster => {
            cluster.entrances?.forEach(ent => {
                selectedEntrances.push({lat: ent.lat, lon: ent.lon, cluster_id: cluster.cluster_id});
            });
        });

        div.remove();
        drawPoints();  // 변경 후 지도 재렌더링
    };

    div.appendChild(confirmBtn);

    document.body.appendChild(div);
}

// 클릭 이벤트 설정
function setupClickEvent() {
    if (mapDiv) {
        mapDiv.on('plotly_click', function(data) {
            if (data && data.points && data.points.length > 0) {
                const point = data.points[0].customdata;
                showEntranceInfo(point);
            }
        });
    } else {
        console.error("mapDiv is null, cannot setup click event");
    }
}

// 전체 IC 목록 CSV 생성
function createICList() {
    const rows = [];
    // 헤더 행
    rows.push(["cluster_id", "IC명", "노선명", "lat", "lon"]);

    clusters_json.forEach(cluster => {
        cluster.entrances?.forEach(ent => {
            rows.push([
                cluster.cluster_id,
                ent.meta?.시설명 || "",
                ent.meta?.노선명 || "",
                ent.lat,
                ent.lon
            ]);
        });
    });

    // CSV 문자열로 변환
    const csvContent = rows.map(r =>
        r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(",")
    ).join("\n");

    // BOM 추가 (Excel 한글 깨짐 방지)
    const bom = "\uFEFF";
    const blob = new Blob([bom + csvContent], { type: "text/csv;charset=utf-8;" });
    return blob; // Blob 자체를 반환
}
//최근점 IC 목록 CSV 생성
function createNearestICList() {
    const rows = [];
    // 헤더 행
    rows.push(["cluster_id", "IC명", "노선명", "lat", "lon"]);

    sortEntrancesByClusterAnchor();
    
    clusters_json.forEach(cluster => {
         const ent = cluster.entrances[0];
            rows.push([
                cluster.cluster_id,
                ent.meta?.시설명 || "",
                ent.meta?.노선명 || "",
                ent.lat,
                ent.lon
            ]);
    });
    // CSV 문자열로 변환
    const csvContent = rows.map(r =>
        r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(",")
    ).join("\n");

    // BOM 추가 (Excel 한글 깨짐 방지)
    const bom = "\uFEFF";
    const blob = new Blob([bom + csvContent], { type: "text/csv;charset=utf-8;" });
    return blob; // Blob 자체를 반환
}

function sortEntrancesByClusterAnchor() {
    [...new Set(clusters_json.map(c => c.cluster_id))].forEach(clusterId => {
        // 해당 cluster_id의 모든 cluster 객체
        const clusterGroup = clusters_json.filter(c => c.cluster_id === clusterId);

        // 1. cluster 내 모든 좌표 후보를 모음
        const candidatePoints = [];
        clusterGroup.forEach(cluster => {
            cluster.points?.forEach(pt => {
                candidatePoints.push({ lat: pt.lat, lon: pt.lon });
            });
        });
        if (candidatePoints.length === 0) return;

        // 2. centroid 계산
        const n = candidatePoints.length;
        const sum = candidatePoints.reduce((acc, e) => {
            acc.lat += e.lat;
            acc.lon += e.lon;
            return acc;
        }, { lat: 0, lon: 0 });
        const centroid = { lat: sum.lat / n, lon: sum.lon / n };

        // 3. centroid에 가장 가까운 실제 후보점 찾기
        let anchor = candidatePoints[0];
        let minDist = haversineDistance(centroid.lat, centroid.lon, anchor.lat, anchor.lon);
        candidatePoints.forEach(e => {
            const d = haversineDistance(centroid.lat, centroid.lon, e.lat, e.lon);
            if (d < minDist) {
                minDist = d;
                anchor = e;
            }
        });

        // 4. clusterGroup 내 엔트리스(entrances) 거리 기준 정렬
        clusterGroup.forEach(cluster => {
            if (cluster.entrances?.length > 0) {
                cluster.entrances.sort((a, b) => {
                    const da = haversineDistance(anchor.lat, anchor.lon, a.lat, a.lon);
                    const db = haversineDistance(anchor.lat, anchor.lon, b.lat, b.lon);
                    return da - db; // 가까운 순
                });
            }
        });
    });
}

// Haversine 거리 계산 함수
function haversineDistance(lat1, lon1, lat2, lon2) {
    const R = 6371e3; // 지구 반지름 (m)
    const toRad = deg => deg * Math.PI / 180;
    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const a = Math.sin(dLat / 2) ** 2 +
              Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
              Math.sin(dLon / 2) ** 2;
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
}

// 다운로드 버튼 설정
function setupDownloadButton() {
    const downloadBtn = document.getElementById('downloadBtn');
    if (!downloadBtn) return;

    downloadBtn.onclick = () => {
        // clusters_json JSON 다운로드
        const jsonBlob = new Blob([JSON.stringify(clusters_json, null, 2)], { type: "application/json;charset=utf-8" });
        const jsonUrl = URL.createObjectURL(jsonBlob);
        const dlJson = document.createElement('a');
        dlJson.href = jsonUrl;
        dlJson.download = regionName + ".json";
        document.body.appendChild(dlJson);
        dlJson.click();
        document.body.removeChild(dlJson);
        URL.revokeObjectURL(jsonUrl);

        // IC CSV 다운로드
        const csvBlob = createNearestICList();
        const csvUrl = URL.createObjectURL(csvBlob);
        const dlCsv = document.createElement('a');
        dlCsv.href = csvUrl;
        dlCsv.download = regionName + ".csv";
        document.body.appendChild(dlCsv);
        dlCsv.click();
        document.body.removeChild(dlCsv);
        URL.revokeObjectURL(csvUrl);
    };
}
// 초기 그리기
// drawPoints();