from geopy.distance import geodesic
import numpy as np
import json
import pandas as pd 


# 1. 정류장별 통과노선 데이터 로드 (https://gits.gg.go.kr/gtdb/web/trafficDb/publicTransport/routeByBusStop.do 여기서 데이터 가져왔습니다.)
transfer_df = pd.read_excel(
    "정류장별_통과노선.xlsx",  
    sheet_name="RouteBybusStop",
    header=1                   # 두 번째 행을 컬럼명으로 사용
)

# '통과노선수' 컬럼이 2 이상인 정류장 ID 리스트 생성
transfer_stop_ids = transfer_df.loc[
    transfer_df["통과노선수"] >= 2, 
    "정류소ID"
].astype(str).tolist()


# 2. 과천시 교통 노드 데이터 로드 
with open("과천시_교통노드.json", "r", encoding="utf-8") as f:
    node_geo = json.load(f)
nodes = [
    {
        "latitude": feat["geometry"]["coordinates"][1],
        "longitude": feat["geometry"]["coordinates"][0],
        "node_type": feat["properties"].get("node_type")
    }
    for feat in node_geo["response"]["result"]["featureCollection"]["features"]
]

# 3. 과천시 교통 링크 데이터 로드 
with open("과천시_교통링크.json", "r", encoding="utf-8") as f:
    link_geo = json.load(f)



def normalize_stop_count(N_stop, N_ideal=30, N_max=15): 
    """
    정류장 수 정규화 함수
    
    N_stop: 현재 노선의 정류장 수
    N_ideal: 이상적인 정류장 수
    N_max: 이상적 정류장 수로부터 허용 가능한 최대 편차 (±N_max)
    
    반환값: 0~1 범위의 정규화 점수 (1에 가까울수록 이상적)
    """
    diff = (N_stop - N_ideal) / N_max
    score = 1 - diff**2
    return max(0.0, min(1.0, score))




def node_alignment_scores(stops, nodes, node_type=None, radius=50, scale=100):   
    """
    교통 노드 정규화 함수
    
    stops: 위도(lat)와 경도(lon) 쌍을 요소로 갖는 리스트  
    nodes: 키를 포함하는 딕셔너리들의 리스트 'latitude','longitude','node_type'
    node_type: 필터할 노드 유형 (None=모든 노드)
    radius: 근접 판정 반경 (m)
    scale: 거리 점수 스케일 값 (m)
    """
    # ① 노드 경유 비율
    near_count = 0
    dists = []
    for lat, lon in stops:
        # 해당 stop이 radius 이내에 node_type 노드가 있으면 카운트
        d2nodes = [
            geodesic((lat,lon),(n['latitude'],n['longitude'])).meters
            for n in nodes
            if node_type is None or n['node_type']==node_type
        ]
        dmin = min(d2nodes) if d2nodes else np.inf
        dists.append(dmin)
        if dmin <= radius:
            near_count += 1

    S_node = near_count / len(stops)
    # ③ 거리 기반 점수
    D_avg = np.mean(dists)
    S_dist = np.exp(-D_avg / scale)

    return S_node, S_dist


def balanced_length_score(length_m, L_ideal=200, L_max=300):
    """
    교통링크 정규화 함수
    
    각 도로 링크의 실제 길이(length_m)가 이상적인 길이(L_ideal)에 얼마나 가까운지를
    정규화하여 0~1 사이의 점수로 반환합니다. 1에 가까울수록 이상 범위에 가까움을 의미합니다.

    Parameters
    ----------
    length_m :
        도로 링크의 실제 길이(m). GeoJSON에서 계산된 거리 값입니다.
    L_ideal : 
        선호하는 이상적인 링크 길이(m). 이 값에 가까울수록 높은 점수를 부여합니다.
    L_max : 
        이상적 길이로부터 허용 가능한 최대 편차(m). 편차가 이 범위를 넘으면 점수는 0으로 수렴합니다.

    Returns
    -------
    score : float
        0.0에서 1.0까지의 정규화된 점수.
        - length_m == L_ideal → 1.0
        - length_m == L_ideal ± L_max → 0.0
        - 그 외에는 1 - diff^2 계산 후 [0,1] 범위로 클리핑(clamp)됩니다.

    Notes
    -----
    - diff = (length_m - L_ideal) / L_max로 정의하며, 편차 비율(diff)이 클수록 벌점이 제곱으로 커집니다.
    - max(0.0, min(1.0, score))를 사용해 결과를 [0,1]로 제한합니다.
    """
    # 이상 길이 대비 실제 길이 편차 비율 계산
    diff = (length_m - L_ideal) / L_max
    # 편차 제곱에 따른 벌점: 이상 길이에 가까울수록 score가 커짐
    score = 1 - diff ** 2
    # 0~1 범위로 클리핑
    return max(0.0, min(1.0, score))


def duplication_penalty_score(weights, counts):
    """
    지역 공정성 적합도 함수 

    Parameters
    ----------
    weights : list of float
        각 요소 i에 대한 중복 가중치 w_dup_i 리스트
    counts : list of int
        각 요소 i에 대한 중복 횟수 count_i 리스트
        weights와 길이가 같아야 함

    Returns
    -------
    float
        0.0 ~ 1.0 사이의 점수.
        - 중복이 전혀 없으면(score = 1.0)
        - 중복이 많아질수록 score는 0에 가까워짐
    """
    if len(weights) != len(counts):
        raise ValueError("weights와 counts의 길이가 같아야 합니다.")

    N = len(weights)
    total_penalty = 0.0
    for w, c in zip(weights, counts):
        total_penalty += w * c

    score = 1.0 - total_penalty / N
    # 결과를 0~1 사이로 클리핑
    return max(0.0, min(1.0, score))


def compute_transfer_score(route_stop_ids, transfer_stop_ids):
    """
    route_stop_ids: 후보 노선이 포함한 정류장 ID 리스트
    transfer_stop_ids: 기존 데이터 기반 환승 정류장 후보 리스트
    '정류소별 통과노선수' 데이터를 활용하여 각 정류장마다 통과하는 노선의 개수를 알 수 있고 값은 1부터 3까지 분포하지만 2개 이상의 노선이 통과하는 정류장을 필터링하여
    함수에 들어갑니다
    """
    if not route_stop_ids:
        return 0.0
    transfer_count = sum(1 for stop in route_stop_ids if stop in transfer_stop_ids)
    return transfer_count / len(route_stop_ids)