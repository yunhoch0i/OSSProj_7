# =========================
# LLM 가중치 및 정류장-POI 병합 함수
# =========================
def load_reference_weights(filepath="llm_response.txt"):
    weights = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":")
                try:
                    weights[key.strip()] = float(val.strip())
                except ValueError:
                    pass
    # F5(w5)는 실질적으로 사용하지 않을 경우 자동 제거
    weights.pop("w5", None)
    return weights

def integrate_stop_poi(bus_stop_json, poi_data):
    """정류장 데이터에 poi_counts 병합"""
    for stop in bus_stop_json:
        stop_id = str(stop.get("정류소id", stop.get("정류소ID", "")))
        matching_poi = next((p for p in poi_data if p["id"] == stop_id), None)
        if matching_poi:
            stop["poi_counts"] = matching_poi["poi_counts"]
        else:
            stop["poi_counts"] = {}
    return bus_stop_json
import networkx as nx  # 그래프 생성 및 경로 탐색
import requests  # Kakao Map API 호출
import json  # JSON 파일 처리
import random  # 유전 알고리즘의 무작위 선택
import numpy as np  # 수치 계산
from sklearn.cluster import KMeans  # 정류장 레이블링을 위한 클러스터링
from heapq import heappush, heappop  # A* 알고리즘의 우선순위 큐
from collections import Counter  # 지역 유형 카운팅
from dotenv import load_dotenv
import os  # 파일 및 환경 변수 처리
from typing import Dict, List, Tuple, Set, Any
from datetime import datetime  # 출력 파일명에 타임스탬프 추가
import time
import osmnx as ox
import pandas as pd
from function_1_to_5 import route_length, stop_distance, poi_score, subway_distance
from function_6_to_10 import (
    normalize_stop_count, node_alignment_scores,
    balanced_length_score, duplication_penalty_score, compute_transfer_score
)
from geopy.distance import geodesic
import math

# 상수 정의
MIN_ROAD_WIDTH = 6  # 최소 도로 폭 (미터)
TARGET_CITY = "Gwacheon"  # 대상 도시: 과천시
MIN_ROAD_WIDTH = 6


# .env 파일에서 환경변수 로드
load_dotenv()

# REST API 키 가져오기
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")

if not KAKAO_REST_API_KEY:
    raise ValueError("KAKAO_REST_API_KEY 환경변수가 없습니다!")

HEADERS = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}



CACHE_FILE = "path_cache.json"

try:
    with open(CACHE_FILE, "r", encoding="utf-8") as _f:
        path_cache = json.load(_f)
except (FileNotFoundError, json.JSONDecodeError):
    # 파일이 없거나 JSON 형식이 잘못된 경우, 빈 dict로 초기화
    print(f"⚠️ 캐시 파일 로드 실패({CACHE_FILE}). 새로운 캐시를 생성합니다.")
    path_cache = {}

# JSON 데이터 로드 함수
# 필요 파일: bus_stop.json, 정류장별_통과노선.json, 과천시_교통노드.json, 과천시_교통링크.json
def load_json_data(filename, encoding="utf-8"):
    """JSON 파일을 로드하고 에러 처리"""
    try:
        with open(filename, "r", encoding=encoding) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"에러: {filename} 파일을 찾을 수 없습니다.")
        return None
    except json.JSONDecodeError as e:
        print(f"에러: {filename} 파일의 JSON 형식이 잘못되었습니다. ({e})")
        return None

# 정류장별 통과노선 데이터 로드
# 파일: 정류장별_통과노선.json (형식: [{"정류소ID": str, "통과노선수": int}, ...])
transfer_data = load_json_data(r"통과노선/정류장별_통과노선.json")
transfer_stop_ids = []
if transfer_data:
    # 환승 가능 정류장(통과노선수 >= 2) ID 추출
    transfer_stop_ids = [str(item["정류소ID"]) for item in transfer_data if item.get("통과노선수", 0) >= 2]

# 교통 노드 데이터 로드
# 파일: 과천시_교통노드.json (형식: GeoJSON, node_type 포함)
node_geo = load_json_data(r"교통노드/과천시_교통노드.json")
nodes = []
if node_geo:
    nodes = [
        {
            "latitude": item["geometry"]["coordinates"][1],
            "longitude": item["geometry"]["coordinates"][0],
            "node_type": item["properties"].get("nd_type_h")
        }
        for item in node_geo["response"]["result"]["featureCollection"]["features"]
    ]

# 교통 링크 데이터 로드
# 파일: 과천시_교통링크.json (형식: GeoJSON, 도로 좌표 포함)
link_geo = load_json_data(r"교통링크/과천시_교통링크.json") or {"response": {"result": {"featureCollection": {"features": []}}}}

# Kakao Map API로 경로 조회
def get_kakao_path(start_lon, start_lat, end_lon, end_lat):
    url = "https://apis-navi.kakaomobility.com/v1/directions"
    params = {
        "origin": f"{start_lon},{start_lat}",
        "destination": f"{end_lon},{end_lat}",
        "priority": "RECOMMEND",
        "car_type": 1,
        "car_fuel": "GASOLINE"
    }
    try:
        resp = requests.get(url, headers=HEADERS, params=params)
        if resp.status_code != 200:
            print("  → 요청 실패 상태 코드:", resp.status_code)
            print("  → 응답 내용:", resp.text)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("routes"):
            return None, 0
        route = data["routes"][0]
        distance = route["summary"]["distance"] / 1000  # m → km
        path_coords = []
        for section in route.get("sections", []):
            for road in section.get("roads", []):
                coords = road.get("vertexes", [])
                for i in range(0, len(coords), 2):
                    path_coords.append((coords[i+1], coords[i]))  # (lat, lon)
        return path_coords, distance
    except requests.RequestException as e:
        print(f"  → 경로 조회 오류: {e}")
        return None, 0







# A* 알고리즘 구현
def astar_path(G, start, goal, constraints=None):
    """A* 알고리즘으로 최단 경로 탐색"""
    if constraints is None:
        constraints = {"min_width": MIN_ROAD_WIDTH}

    def heuristic(node1, node2, G):
        """두 노드 간 직선 거리(지오데식 거리) 계산"""
        try:
            # 노드 존재 여부 확인
            if node1 not in G.nodes or node2 not in G.nodes:
                raise ValueError(f"노드 {node1} 또는 {node2}가 그래프에 없습니다.")
            pos1 = G.nodes[node1].get("pos")
            pos2 = G.nodes[node2].get("pos")
            # 좌표 유효성 검사
            if pos1 is None or pos2 is None:
                raise ValueError(f"노드 {node1} 또는 {node2}에 'pos' 속성이 없습니다.")
            lon1, lat1 = pos1
            lon2, lat2 = pos2
            if not all(isinstance(coord, (int, float)) for coord in [lon1, lat1, lon2, lat2]):
                raise ValueError(f"노드 {node1} ({pos1}) 또는 {node2} ({pos2})의 좌표가 유효하지 않습니다.")
            # 지오데식 거리 계산 (km)
            return geodesic((lat1, lon1), (lat2, lon2)).km
        except Exception as e:
            print(f"heuristic 계산 중 에러 (노드 {node1}, {node2}): {e}")
            return float('inf')  # 유효하지 않은 경로에 큰 거리 반환

    # A* 알고리즘 초기화
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal, G)}

    while open_set:
        current_f, current = heappop(open_set)

        if current == goal:
            # 경로 재구성
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        # 이웃 노드 탐색
        for neighbor in G.neighbors(current):
            edge_data = G[current][neighbor]
            if edge_data.get("width", MIN_ROAD_WIDTH) < constraints.get("min_width", 0):
                continue

            tentative_g_score = g_score[current] + edge_data["weight"]

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal, G)
                heappush(open_set, (f_score[neighbor], neighbor))

    print(f"{start}에서 {goal}로 가는 경로를 찾을 수 없습니다.")
    return None

# 정류장 데이터 로드
# 파일: bus_stop.json (형식: [{"정류소ID": str, "정류장명": str, "위도(WGS84)": float, "경도(WGS84)": float, "passengers": float, "population": float}, ...])

def load_bus_stops(json_data):
    """
    JSON 형식의 정류장 데이터 로드 및 '같은 좌표' 중복 제거
    """
    stops = []
    seen_coords = set()

    for i, row in enumerate(json_data):
        try:
            lat = float(row["WGS84위도"])
            lon = float(row["WGS84경도"])
        except (KeyError, ValueError):
            print(f"경고: 정류장 {i}의 위도/경도 변환 실패 → 스킵")
            continue

        coord = (round(lat, 7), round(lon, 7))
        if coord in seen_coords:
            continue
        seen_coords.add(coord)

        name = row.get("정류소명", "").strip()
        if not name:
            print(f"경고: 정류장 {i} - 이름 없음 → 스킵")
            continue

        stop_id = "stop_" + str(row.get("정류소id", f"{name}_{i}"))
        passengers = float(row.get("passengers", 100)) if row.get("passengers") else 100
        population = float(row.get("population", 500)) if row.get("population") else 500

        stops.append({
            "id": stop_id,
            "name": name,
            "lat": lat,
            "lon": lon,
            "passengers": passengers,
            "population": population
        })

    print(f"로드된 정류장 수 (중복 좌표 제거 후): {len(stops)}")
    return stops


def fetch_kakao_road_data(stops):
    """Kakao Map API를 사용해 정류장 간 도로 네트워크 생성 (필터링 완화 버전)"""
    roads = []
    G_road = nx.Graph()
    global path_cache

    for i, start_stop in enumerate(stops):
        start_id = str(start_stop["id"])
        G_road.add_node(start_id, pos=(start_stop["lon"], start_stop["lat"]), name=start_stop["name"])

        for j, end_stop in enumerate(stops[i+1:], start=i+1):
            end_id = str(end_stop["id"])
            # 좌표 유효성 검사
            if not all(isinstance(coord, (int, float)) for coord in [end_stop["lon"], end_stop["lat"]]):
                continue

            # 직선 거리 계산
            raw_dist = geodesic((start_stop["lat"], start_stop["lon"]),
                                (end_stop["lat"], end_stop["lon"])).km
            # ← 여기서 0.3km 대신 예컨대 1.0km로 변경
            if raw_dist > 1.0:
                # 너무 먼 쌍은 일단 스킵
                continue

            print(f"[{i},{j}] 요청 중: {start_stop['name']} → {end_stop['name']} (직선 거리 = {raw_dist:.3f} km)")

            cache_key = f"{start_id}_{end_id}"
            if cache_key in path_cache:
                cached = path_cache[cache_key]
                path_coords = cached["path_coords"]
                distance = cached["distance"]
            else:
                time.sleep(0.3)
                path_coords, distance = get_kakao_path(
                    start_stop["lon"], start_stop["lat"],
                    end_stop["lon"], end_stop["lat"]
                )
                # API 항상 성공하지 않으므로 None 체크
                if path_coords and distance > 0:
                    path_cache[cache_key] = {"path_coords": path_coords, "distance": distance}
                    with open(CACHE_FILE, "w", encoding="utf-8") as _cf:
                        json.dump(path_cache, _cf, ensure_ascii=False, indent=2)
            if path_coords and distance > 0:
                G_road.add_edge(start_id, end_id, weight=distance, width=MIN_ROAD_WIDTH)
                roads.append({
                    "id": f"road_{start_id}_{end_id}",
                    "start": {"lat": start_stop["lat"], "lon": start_stop["lon"]},
                    "end": {"lat": end_stop["lat"], "lon": end_stop["lon"]},
                    "length": distance,
                    "width": MIN_ROAD_WIDTH,
                    "path_coords": path_coords
                })

    return {"roads": roads, "road_graph": G_road}



# 노드 레이블링
def label_nodes(pois, stops, k=3):
    """POI와 정류장 데이터를 사용해 정류장에 지역 유형 레이블 부여"""
    if not pois or len(pois) < k:
        k = max(2, len(pois)) if pois else 2

    # POI가 없으면 정류장 좌표 사용
    poi_coords = np.array([[poi["lat"], poi["lon"]] for poi in pois]) if pois else np.array([[stop["lat"], stop["lon"]] for stop in stops])
    poi_types = [poi["type"] for poi in pois] if pois else ["unknown"] * len(stops)

    # K-means 클러스터링
    kmeans = KMeans(n_clusters=k, random_state=0).fit(poi_coords)

    # 클러스터별 주요 POI 유형 결정
    cluster_types = {}
    for i in range(k):
        cluster_pois = [poi_types[j] for j in range(len(poi_types)) if kmeans.labels_[j] == i]
        most_common = Counter(cluster_pois).most_common(1)
        cluster_types[i] = most_common[0][0] if most_common else "기타"

    # 지역 유형 매핑
    label_mapping = {
        "school": "교육지구",
        "hospital": "의료지구",
        "restaurant": "상업지구",
        "cafe": "상업지구",
        "shop": "상업지구",
        "park": "여가지구",
        "residential": "주거지역",
        "office": "업무지구"
    }

    for i in cluster_types:
        if cluster_types[i] in label_mapping:
            cluster_types[i] = label_mapping[cluster_types[i]]

    # 정류장에 레이블 부여
    stop_labels = {}
    for stop in stops:
        nearby_pois = [i for i, poi in enumerate(pois)
                       if geodesic((stop["lat"], stop["lon"]), (poi["lat"], poi["lon"])).km <= 0.3] if pois else []
        if nearby_pois:
            cluster = Counter(kmeans.labels_[nearby_pois]).most_common(1)[0][0]
            stop_labels[stop["id"]] = cluster_types[cluster]
        else:
            stop_labels[stop["id"]] = "기타"

    return stop_labels

# 도시 그래프 생성
def create_city_graph(stops, pois, roads, road_graph):
    """정류장, POI, 도로 데이터를 사용해 도시 그래프 생성"""
    G = nx.Graph()
    stop_labels = label_nodes([], stops)

    # 정류장 노드 추가 (POI 정보 포함)
    for stop in stops:
        if not all(isinstance(coord, (int, float)) for coord in [stop["lon"], stop["lat"]]):
            print(f"경고: 정류장 {stop['id']}의 좌표가 유효하지 않습니다: ({stop['lon']}, {stop['lat']})")
            continue


        G.add_node(stop["id"],
                   pos=(stop["lon"], stop["lat"]),
                   name=stop["name"],
                   label=stop_labels[stop["id"]],
                   passengers=stop.get("passengers", 0),
                   population=stop.get("population", 0),
                   poi_counts=stop.get("poi_counts", {}))

    # 도로 엣지 추가
    for u, v, data in road_graph.edges(data=True):
        if u not in G.nodes or v not in G.nodes:
            print(f"경고: 엣지 ({u}, {v})가 존재하지 않는 노드를 참조합니다.")
            continue
        G.add_edge(u, v, weight=data["weight"], width=data.get("width", MIN_ROAD_WIDTH))

    print(f"그래프 노드 수: {G.number_of_nodes()}, 엣지 수: {G.number_of_edges()}")
    isolated = list(nx.isolates(G))
    if isolated:
        print(f"연결되지 않은 노드: {isolated}")

    return G

# 유전 알고리즘 적합도 함수
# 필요 변수: reference_weights (적합도 가중치 및 기준값)
def fitness(route, G, reference_weights, stops_data, pois_data, nodes_data, link_data, transfer_ids):
    """노선의 적합도를 F1~F10의 가중 합으로 계산"""
    if len(route) < 2:
        return 0

    # (1) F1 ~ F2: 경로 길이, 정류장 간 거리 계산
    length = 0
    paths = []
    for i in range(len(route) - 1):
        path = astar_path(G, route[i], route[i + 1])
        if path and len(path) > 1:
            path_length = sum(G[path[j]][path[j + 1]]["weight"] for j in range(len(path) - 1))
            length += path_length
            paths.append(path)
        else:
            return 0  # 유효한 경로가 아니면 적합도 0

    # F1: route_length
    L_i = reference_weights.get("avg_length", 5.0)
    L_max = reference_weights.get("max_length", 7.5)
    f1 = route_length(length, L_i, L_max)

    # F2: stop_distance (직선 거리 기준)
    stop_coords = [(G.nodes[stop]["pos"][1], G.nodes[stop]["pos"][0]) for stop in route]
    distances = [
        geodesic((stop_coords[i][0], stop_coords[i][1]), (stop_coords[i+1][0], stop_coords[i+1][1])).km
        for i in range(len(stop_coords) - 1)
    ]
    D_ideal = reference_weights.get("avg_distance", 0.7)
    sigma = reference_weights.get("sigma", 0.2)
    f2 = stop_distance(distances, D_ideal, sigma)

    # ─────────────────────────────────────────
    # (2) F3: poi_score (정류장별 poi_counts 사용)
    poi_list = []
    weight_dict = reference_weights.get("poi_weights", {
        "school": 0.3, "hospital": 0.3, "restaurant": 0.2, "cafe": 0.2,
        "shop": 0.2, "park": 0.15, "residential": 0.1, "office": 0.1
    })

    for stop in route:
        counts = G.nodes[stop].get("poi_counts", {})
        for poi_type, cnt in counts.items():
            if cnt and cnt > 0:
                poi_list.append({"type": poi_type, "count": cnt})

    f3 = poi_score(poi_list, weight_dict) if poi_list else 0
    # ─────────────────────────────────────────

    # F4: subway_distance
    subway_nodes = [n for n in nodes_data if n.get("node_type") == "subway"]
    subway_dists = []
    for stop in route:
        lat, lon = G.nodes[stop]["pos"][1], G.nodes[stop]["pos"][0]
        dists = [geodesic((lat, lon), (n["latitude"], n["longitude"])).meters for n in subway_nodes]
        if dists:
            subway_dists.append(min(dists))
    D_scale = reference_weights.get("D_scale", 1000)
    f4 = subway_distance(subway_dists, D_scale) if subway_dists else 0

    # F5: 유동인구 (데이터 없음 → 0)
    f5 = 0

    # F6: normalize_stop_count
    N_ideal = reference_weights.get("avg_stop_count", 12)
    N_max = reference_weights.get("max_stop_deviation", 5)
    f6 = normalize_stop_count(len(route), N_ideal, N_max)

    # F7, F8: node_alignment_scores (예시: 지하철 노드와 정렬 점수)
    stops_coords = [(G.nodes[stop]["pos"][1], G.nodes[stop]["pos"][0]) for stop in route]
    f7, f8 = node_alignment_scores(stops_coords, nodes_data, node_type="subway", radius=50, scale=100)

    # F9: balanced_length_score (도로 링크 정보 활용)
    link_lengths = []
    if isinstance(link_data, list):
        features = link_data
    else:
        features = link_data.get("response", {}).get("result", {}).get("featureCollection", {}).get("features", [])

    for feat in features:
        if "geometry" not in feat or "coordinates" not in feat["geometry"]:
            continue
        coords = feat["geometry"]["coordinates"]
        if len(coords) >= 2:
            length_m = sum(geodesic(coords[i], coords[i + 1]).meters for i in range(len(coords) - 1))
            link_lengths.append(length_m)

    L_ideal = reference_weights.get("L_ideal", 200)
    L_max = reference_weights.get("L_max", 300)
    f9 = np.mean([balanced_length_score(l, L_ideal, L_max) for l in link_lengths]) if link_lengths else 0

    # F10: compute_transfer_score
    route_stop_ids = [stop for stop in route if "stop" in stop]
    f10 = compute_transfer_score(route_stop_ids, transfer_ids)

    # (3) 가중 합
    weights = reference_weights.get("fitness_weights", {
        "w1": 0.100, "w2": 0.100, "w3": 0.150, "w4": 0.100, "w5": 0.050,
        "w6": 0.100, "w7": 0.050, "w8": 0.150, "w9": 0.100, "w10": 0.100
    })
    fitness_score = (
        weights["w1"] * f1 +
        weights["w2"] * f2 +
        weights["w3"] * f3 +
        weights["w4"] * f4 +
        weights["w5"] * f5 +
        weights["w6"] * f6 +
        weights["w7"] * f7 +
        weights["w8"] * f8 +
        weights["w9"] * f9 +
        weights["w10"] * f10
    )

    return max(0, fitness_score)

# 유전 알고리즘
def genetic_algorithm(G, reference_weights, num_routes, stops_data, pois_data, nodes_data, link_data, transfer_ids, population_size=50, generations=100):
    """유전 알고리즘으로 최적 버스 노선 생성"""
    all_stops = [n for n in G.nodes() if isinstance(n, str) and "stop" in n]
    if len(all_stops) < 3:
        print("에러: 그래프에 정류장이 충분하지 않습니다.")
        return []

    # 승객 수가 가장 많은 정류장을 허브로 선택
    hub = max(all_stops, key=lambda x: G.nodes[x]["passengers"])
    target_stop_count = int(reference_weights.get("avg_stop_count", 12))

    # 초기 노선 집합 생성
    population = []
    for _ in range(population_size):
        route = [hub]
        remaining = set(all_stops) - {hub}
        attempts = 0
        max_attempts = len(all_stops) * 3
        while len(route) < target_stop_count and remaining and attempts < max_attempts:
            candidates = list(remaining)
            random.shuffle(candidates)
            for next_stop in candidates:
                path = astar_path(G, route[-1], next_stop)
                if path and len(path) > 1:
                    for stop in path[1:]:
                        if stop not in route and "stop" in stop:
                            route.append(stop)
                            if stop in remaining:
                                remaining.remove(stop)
                    break
            else:
                if candidates:
                    remaining.remove(candidates[0])
            attempts += 1
        if len(route) >= 3:
            population.append(list(dict.fromkeys(route)))

    if not population:
        print("에러: 유효한 초기 노선을 생성하지 못했습니다.")
        return []

    # 유전 알고리즘 반복
    for gen in range(generations):
        scored_population = [(route, fitness(route, G, reference_weights, stops_data, pois_data, nodes_data, link_data, transfer_ids)) for route in population]
        scored_population.sort(key=lambda x: x[1], reverse=True)
        # 선택: fitness를 가중치로 사용하여 selection_size 만큼 선택
        selection_size = population_size // 2
        # fitness가 모두 0일 경우 대비
        fitness_values = [max(0, score) for _, score in scored_population]
        # 선택 확률이 모두 0이면 균등 선택
        if sum(fitness_values) == 0:
            selected = random.choices([route for route, _ in scored_population], k=selection_size)
        else:
            selected = random.choices(
                [route for route, _ in scored_population],
                weights=fitness_values,
                k=selection_size
            )
        new_population = list(selected)

        # 교차 및 돌연변이
        while len(new_population) < population_size:
            # 부모 선택 (fitness 기반 selection에서 이미 선택된 개체 중 무작위 2개)
            parent1, parent2 = random.sample(selected, 2)
            child = parent1.copy()
            if len(parent1) > 2 and len(parent2) > 2:
                crossover_point = random.randint(1, min(len(parent1), len(parent2))-1)
                child = parent1[:crossover_point]
                child.extend([s for s in parent2 if s not in child])

            if random.random() < 0.1 and len(child) > 0 and len(all_stops) > len(child):
                mutation_idx = random.randint(0, len(child)-1)
                new_stop = random.choice(list(set(all_stops) - set(child)))
                if mutation_idx > 0 and mutation_idx < len(child) - 1:
                    prev_stop = child[mutation_idx - 1]
                    next_stop = child[mutation_idx + 1]
                    path1 = astar_path(G, prev_stop, new_stop)
                    path2 = astar_path(G, new_stop, next_stop)
                    if path1 and path2:
                        child[mutation_idx] = new_stop
                elif mutation_idx == 0:
                    next_stop = child[mutation_idx + 1]
                    if astar_path(G, new_stop, next_stop):
                        child[mutation_idx] = new_stop
                elif mutation_idx == len(child) - 1:
                    prev_stop = child[mutation_idx - 1]
                    if astar_path(G, prev_stop, new_stop):
                        child[mutation_idx] = new_stop

            # 유효한 경로 검증
            valid_route = [child[0]]
            for i in range(1, len(child)):
                path = astar_path(G, valid_route[-1], child[i])
                if path:
                    for stop in path[1:]:
                        if stop not in valid_route and "stop" in stop:
                            valid_route.append(stop)
            if len(valid_route) >= 3:
                if len(valid_route) > target_stop_count:
                    valid_route = valid_route[:target_stop_count]
                new_population.append(valid_route)

        population = new_population
        if (gen + 1) % 10 == 0:
            best_score = scored_population[0][1]
            avg_score = np.mean([score for _, score in scored_population])
            print(f"세대 {gen+1}/{generations}: 최고 점수 = {best_score:.2f}, 평균 점수 = {avg_score:.2f}")

    # 최종 노선 선택
    final_scored_population = [(route, fitness(route, G, reference_weights, stops_data, pois_data, nodes_data, link_data, transfer_ids)) for route in population]
    final_scored_population.sort(key=lambda x: x[1], reverse=True)
    return [route for route, _ in final_scored_population[:num_routes]]

# 버스 배차 계획
def schedule_buses(routes, G, num_buses):
    """노선을 버스에 배정하여 스케줄 생성"""
    route_stats = []
    for i, route in enumerate(routes):
        if len(route) < 2:
            continue
        length = 0
        for j in range(len(route) - 1):
            path = astar_path(G, route[j], route[j + 1])
            if path and len(path) > 1:
                path_length = sum(G[path[k]][path[k + 1]]["weight"] for k in range(len(path) - 1))
                length += path_length
            else:
                continue
        passengers = sum(G.nodes[stop]["passengers"] for stop in route)
        round_trip_time = (length * 2) / 30 * 60  # 왕복 시간 (분, 평균 속도 30km/h)
        route_stats.append({
            "route_id": i,
            "route": route,
            "length": length,
            "passengers": passengers,
            "round_trip_time": round_trip_time
        })

    # 노선을 왕복 시간 기준으로 정렬
    route_stats.sort(key=lambda x: x["round_trip_time"], reverse=True)
    schedule = {f"Bus {i+1}": [] for i in range(num_buses)}
    # 노선을 버스에 균등 배정
    for route_info in route_stats:
        bus_times = {bus: sum(r["round_trip_time"] for r in schedule[bus]) for bus in schedule}
        target_bus = min(bus_times, key=bus_times.get)
        schedule[target_bus].append(route_info)
    return schedule

# 배차 간격 계산
def calculate_frequencies(schedule):
    """각 노선의 배차 간격 계산"""
    frequencies = {}
    for bus, routes in schedule.items():
        for route_info in routes:
            route_id = route_info["route_id"]
            bus_count = sum(1 for b, rs in schedule.items() for r in rs if r["route_id"] == route_id)
            frequency = max(5, min(60, route_info["round_trip_time"] / bus_count))
            frequencies[route_id] = round(frequency)
    return frequencies

# Kakao Map 시각화
def display_routes(G, routes, schedule, frequencies, roads_data, kakao_api_key=None):
    """노선과 정류장을 Kakao Map에 시각화하여 HTML 파일로 저장"""
    pos = nx.get_node_attributes(G, "pos")
    lats = [G.nodes[node]["pos"][1] for node in G.nodes() if "stop" in node]
    lons = [G.nodes[node]["pos"][0] for node in G.nodes() if "stop" in node]
    center_lat = sum(lats) / len(lats) if lats else 37.43
    center_lon = sum(lons) / len(lons) if lons else 126.99

    route_colors = ['#FF0000', '#0000FF', '#008000', '#800080', '#FFA500', '#8B0000', '#00008B', '#006400', '#5F9EA0', '#9932CC']
    map_data = {
        "center": {"lat": center_lat, "lng": center_lon},
        "zoom": 14,
        "stops": [],
        "routes": []
    }

    # 정류장 데이터 추가
    for node in G.nodes():
        if "stop" not in node:
            continue
        lon, lat = pos[node]
        name = G.nodes[node].get("name", node)
        label = G.nodes[node].get("label", "기타")
        passengers = G.nodes[node].get("passengers", 0)
        color_map = {
            "상업지구": "red",
            "주거지역": "blue",
            "교육지구": "green",
            "업무지구": "purple",
            "여가지구": "orange",
            "교통중심지": "black",
            "기타": "gray"
        }
        icon_color = color_map.get(label, "gray")
        map_data["stops"].append({
            "id": node,
            "name": name,
            "lat": lat,
            "lng": lon,
            "label": label,
            "passengers": passengers,
            "color": icon_color
        })

    # 노선 데이터 추가
    for i, route in enumerate(routes):
        route_id = i
        route_color = route_colors[i % len(route_colors)]
        frequency = frequencies.get(route_id, 0)
        route_coords = []
        for j in range(len(route) - 1):
            start = route[j]
            end = route[j + 1]
            road = next((r for r in roads_data if r["id"] == f"road_{start}_{end}"), None)
            if road and road.get("path_coords"):
                route_coords.extend(road["path_coords"])
        map_data["routes"].append({
            "route_id": route_id + 1,
            "coordinates": route_coords,
            "color": route_color,
            "frequency": frequency
        })

    # HTML 템플릿
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Bus Routes Map</title>
        <script type="text/javascript" src="https://dapi.kakao.com/v2/maps/sdk.js?appkey={KAKAO_API_KEY}"></script>
        <style>
            #map {{ width: 100%; height: 800px; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            var mapContainer = document.getElementById('map');
            var mapOption = {{
                center: new kakao.maps.LatLng({center_lat}, {center_lng}),
                level: {zoom}
            }};
            var map = new kakao.maps.Map(mapContainer, mapOption);

            var stops = {stops_json};
            stops.forEach(function(그만) {{
                var markerPosition = new kakao.maps.LatLng(stop.lat, stop.lng);
                var marker = new kakao.maps.Marker({{
                    position: markerPosition,
                    title: stop.name + ' (' + stop.label + ')\\n승객: ' + stop.passengers + '/일'
                }});
                marker.setMap(map);
            }});

            var routes = {routes_json};
            routes.forEach(function(route) {{
                var path = route.coordinates.map(function(coord) {{
                    return new kakao.maps.LatLng(coord[0], coord[1]);
                }});
                var polyline = new kakao.maps.Polyline({{
                    path: path,
                    strokeWeight: 4,
                    strokeColor: route.color,
                    strokeOpacity: 0.8,
                    strokeStyle: 'solid'
                }});
                polyline.setMap(map);

                var infowindow = new kakao.maps.InfoWindow({{
                    content: '노선 ' + route.route_id + ' (배차간격: ' + route.frequency + '분)'
                }});
                kakao.maps.event.addListener(polyline, 'mouseover', function() {{
                    infowindow.open(map, polyline);
                }});
                kakao.maps.event.addListener(polyline, 'mouseout', function() {{
                    infowindow.close();
                }});
            }});
        </script>
    </body>
    </html>
    """

    # HTML 파일 생성
    html_content = html_content.format(
        KAKAO_API_KEY=kakao_api_key or "eba983b6e7f15a4e7c0dcbbd0b47b1dc",  # JavaScript API 키 사용
        center_lat=center_lat,
        center_lng=center_lon,
        zoom=map_data["zoom"],
        stops_json=json.dumps(map_data["stops"]),
        routes_json=json.dumps(map_data["routes"])
    )

    output_file = f"bus_routes.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"지도는 '{output_file}' 파일로 저장되었습니다.")
    return html_content


# 노선 특성 계산
def calculate_route_characteristics(routes, G):
    """노선의 주요 특성(길이, 정류장 수, 지역 유형 등) 계산"""
    characteristics = []
    for i, route in enumerate(routes):
        if len(route) < 2:
            continue
        length = 0
        for j in range(len(route) - 1):
            path = astar_path(G, route[j], route[j + 1])
            if path and len(path) > 1:
                path_length = sum(G[path[k]][path[k + 1]]["weight"] for k in range(len(path) - 1))
                length += path_length
            else:
                continue
        stop_count = len(route)
        area_types = Counter([G.nodes[stop]["label"] for stop in route])
        passengers = sum(G.nodes[stop]["passengers"] for stop in route)
        avg_passengers = passengers / stop_count if stop_count > 0 else 0
        avg_distance = length / (stop_count - 1) if stop_count > 1 else 0
        characteristics.append({
            "route_id": i,
            "stops": [G.nodes[stop].get("name", stop) for stop in route],
            "length": length,
            "stop_count": stop_count,
            "area_types": dict(area_types),
            "total_passengers": passengers,
            "avg_passengers": avg_passengers,
            "avg_distance": avg_distance
        })
    return characteristics

# 결과 요약 출력
def print_route_summary(routes, G, schedule, frequencies, reference_weights=None):
    """생성된 노선과 배차 계획 요약 출력"""
    print("\n=== 버스 노선 요약 ===")
    for i, route in enumerate(routes):
        if len(route) < 2:
            continue
        length = 0
        for j in range(len(route) - 1):
            path = astar_path(G, route[j], route[j + 1])
            if path and len(path) > 1:
                path_length = sum(G[path[k]][path[k + 1]]["weight"] for k in range(len(path) - 1))
                length += path_length
            else:
                continue
        passengers = sum(G.nodes[stop]["passengers"] for stop in route)
        stops = [G.nodes[stop].get("name", stop) for stop in route]
        print(f"\n노선 {i+1}:")
        print(f"  - 정류장 ({len(route)}개): {' -> '.join(stops)}")
        print(f"  - 총 길이: {length:.2f} km")
        print(f"  - 예상 이용객: {passengers}/일")
        print(f"  - 배차 간격: {frequencies.get(i, 0)}분")

    print("\n=== 버스 배차 요약 ===")
    for bus, routes_info in schedule.items():
        if not routes_info:
            continue
        print(f"\n{bus}:")
        for route_info in routes_info:
            route_id = route_info["route_id"]
            print(f"  - 노선 {route_id+1} (배차간격: {frequencies.get(route_id, 0)}분)")

    if reference_weights:
        print("\n=== 참조 가중치와 생성된 노선 비교 ===")
        generated_lengths = []
        generated_stop_counts = []
        generated_avg_distance = []
        generated_passengers = []
        area_type_counts = Counter()
        for route in routes:
            if len(route) < 2:
                continue
            length = 0
            for j in range(len(route) - 1):
                path = astar_path(G, route[j], route[j + 1])
                if path and len(path) > 1:
                    path_length = sum(G[path[k]][path[k + 1]]["weight"] for k in range(len(path) - 1))
                    length += path_length
            if length == 0:
                continue
            generated_lengths.append(length)
            generated_stop_counts.append(len(route))
            avg_distance = length / (len(route) - 1) if len(route) > 1 else 0
            generated_avg_distance.append(avg_distance)
            route_passengers = sum(G.nodes[stop]["passengers"] for stop in route) / len(route)
            generated_passengers.append(route_passengers)
            for stop in route:
                area_type_counts[G.nodes[stop]["label"]] += 1

        if generated_lengths:
            print(f"노선 평균 길이: {np.mean(generated_lengths):.2f} km (참조: {reference_weights.get('avg_length', 5.0):.2f} km)")
            print(f"노선 평균 정류장 수: {np.mean(generated_stop_counts):.2f} (참조: {reference_weights.get('avg_stop_count', 12.0):.2f})")
            print(f"정류장 간 평균 거리: {np.mean(generated_avg_distance):.2f} km (참조: {reference_weights.get('avg_distance', 0.7):.2f} km)")
            print(f"정류장 평균 이용객 수: {np.mean(generated_passengers):.2f}/일 (참조: {reference_weights.get('avg_passengers', 200.0):.2f}/일)")
            print("\n지역 유형 분포:")
            total_stops = sum(area_type_counts.values())
            for area_type, count in area_type_counts.most_common():
                generated_ratio = count / total_stops
                reference_ratio = reference_weights.get("area_type_distribution", {}).get(area_type, 0)
                print(f"  - {area_type}: {generated_ratio:.2%} (참조: {reference_ratio:.2%})")

def load_poi_data(filename="bus_stop.csv"):
    """POI 정보가 포함된 CSV 파일을 로드"""
    try:
        poi_df = pd.read_csv(filename, encoding='utf-8-sig')
        poi_data = []
        for _, row in poi_df.iterrows():
            poi_info = {
                "id": str(row["정류소ID"]),
                "name": row["정류소명"],
                "lat": row["위도"],
                "lon": row["경도"],
                "poi_counts": {
                    "대형마트": row["POI_대형마트"],
                    "편의점": row["POI_편의점"],
                    "어린이집": row["POI_어린이집"],
                    "학교": row["POI_학교"],
                    "학원": row["POI_학원"],
                    "주차장": row["POI_주차장"],
                    "주유소": row["POI_주유소"],
                    "지하철역": row["POI_지하철역"],
                    "은행": row["POI_은행"],
                    "문화시설": row["POI_문화시설"],
                    "중개업소": row["POI_중개업소"],
                    "공공기관": row["POI_공공기관"],
                    "관광명소": row["POI_관광명소"],
                    "숙박": row["POI_숙박"],
                    "음식점": row["POI_음식점"],
                    "카페": row["POI_카페"],
                    "병원": row["POI_병원"],
                    "약국": row["POI_약국"]
                }
            }
            poi_data.append(poi_info)
        return poi_data
    except Exception as e:
        print(f"POI 데이터 로드 중 오류 발생: {e}")
        return []

def generate_bus_routes(reference_weights, target_city, num_routes, num_buses, bus_stop_json):
    """버스 노선 생성 및 시각화 메인 함수"""
    print(f"\n1. {target_city}의 교통 네트워크 구축 중...")
    stops = load_bus_stops(bus_stop_json)

    # ────────── 전역 캐시 사용 선언 ──────────
    global path_cache

    # (Optional) 도로 그래프용 노드는 이 단계에서 바로 만들어도 되고,
    # 아니면 나중에 create_city_graph에서 처리해도 된다.
    G_road = nx.Graph()
    roads = []

    # 1-1) 각 정류장 쌍마다 경로 데이터를 가져와서 G_road에 추가하고, 캐시에 저장한다.
    for i, start_stop in enumerate(stops):
        start_id = str(start_stop["id"])
        # 정류장 노드로 추가 (create_city_graph 단계로 미뤄도 상관 없음)
        G_road.add_node(start_id,
                        pos=(start_stop["lon"], start_stop["lat"]),
                        name=start_stop["name"])

        for j, end_stop in enumerate(stops[i+1:], start=i+1):
            end_id = str(end_stop["id"])

            # 좌표 유효성 검사
            if not all(isinstance(coord, (int, float)) for coord in [end_stop["lon"], end_stop["lat"]]):
                print(f"경고: 정류장 {end_id}의 좌표가 유효하지 않습니다.")
                continue

            # ─────── (A) 직선 거리 계산 & 로그 출력 ───────
            raw_dist_km = geodesic(
                (start_stop["lat"], start_stop["lon"]),
                (end_stop["lat"], end_stop["lon"])
            ).km
            print(f"[{i},{j}]    ▶ 직선 거리 = {raw_dist_km:.3f}km  |  "
                  f"{start_stop['name']} → {end_stop['name']}")

            # ─────── (B) 0.3km 초과면 실제 API 호출 생략 ───────
            if raw_dist_km > 1.0:
                continue  # 0.3km 이상 떨어진 정류장 쌍은 패스

            # ─────── (C) 캐시 키 생성 & 캐시 조회 ───────
            cache_key = f"{start_id}_{end_id}"
            if cache_key in path_cache:
                cached = path_cache[cache_key]
                path_coords = cached.get("path_coords")
                distance = cached.get("distance")
            else:
                # 캐시에 없으면 실제 API 호출
                time.sleep(0.2)  # (권장) 호출 간 짧은 대기
                path_coords, distance = get_kakao_path(
                    start_stop["lon"], start_stop["lat"],
                    end_stop["lon"], end_stop["lat"]
                )
                if path_coords is not None and distance is not None:
                    path_cache[cache_key] = {
                        "path_coords": path_coords,
                        "distance": distance
                    }
                    # 즉시 캐시 파일에 덮어쓰기(영구 저장)
                    with open(CACHE_FILE, "w", encoding="utf-8") as _cf:
                        json.dump(path_cache, _cf, ensure_ascii=False, indent=2)

            # ─────── (D) G_road 와 roads 리스트에 추가 ───────
            if path_coords and distance > 0:
                G_road.add_edge(start_id, end_id,
                                weight=distance,
                                width=MIN_ROAD_WIDTH)
                roads.append({
                    "id":         f"road_{start_id}_{end_id}",
                    "start":      {"lat": start_stop["lat"], "lon": start_stop["lon"]},
                    "end":        {"lat": end_stop["lat"],     "lon": end_stop["lon"]},
                    "length":     distance,
                    "width":      MIN_ROAD_WIDTH,
                    "path_coords": path_coords
                })

    # 1-2) 이제 G_road와 roads 리스트가 완성되었으므로,
    #       create_city_graph 단계에서 사용하기 위해 변수로 전달한다.
    pois = load_poi_data()  # POI는 별도로 로드
    target_graph = create_city_graph(stops, pois, roads, G_road)

    # ────────────────────────────────────────────────
    # 이하 기존 코드: genetic_algorithm → schedule_buses → display_routes 등
    # ────────────────────────────────────────────────
    print(f"\n2. {target_city}에 최적 버스 노선 생성 중...")
    routes = genetic_algorithm(target_graph, reference_weights,
                               num_routes, stops, pois, nodes, link_geo, transfer_stop_ids)

    print(f"\n3. 버스 배차 계획 수립 중...")
    schedule = schedule_buses(routes, target_graph, num_buses)
    frequencies = calculate_frequencies(schedule)

    print_route_summary(routes, target_graph, schedule, frequencies, reference_weights)
    print(f"\n4. 노선도 생성 중...")
    map_html = display_routes(target_graph, routes, schedule, frequencies, roads)

    return routes, schedule, frequencies, map_html





# 메인 함수 (명령줄 인터페이스)
def main():
    print("===== 버스 노선 생성 시스템 =====")
    print(f"대상 도시: {TARGET_CITY}")

    # 1. llm_response.txt에서 가중치 자동로드
    reference_weights = load_reference_weights("llm_response.txt")

    # 2. 입력 데이터 로드
    bus_stop_json = load_json_data("정류장위치/과천시_버스_정류장_위치.json")
    if not bus_stop_json:
        print("에러: bus_stop.json 파일을 로드하지 못했습니다.")
        return

    poi_data = load_poi_data("bus_stop.csv")
    if not poi_data:
        print("경고: POI 데이터를 로드하지 못했습니다. POI 미반영으로 진행.")

    # 3. 정류장-POI 데이터 통합
    bus_stop_json = integrate_stop_poi(bus_stop_json, poi_data)

    # 4. 사용자 입력
    num_routes = int(input("생성할 노선 개수를 입력하세요: "))
    num_buses = int(input("운행 가능한 버스 대수를 입력하세요: "))

    # 5. 버스 노선 생성
    routes, schedule, frequencies, map_html = generate_bus_routes(
        reference_weights, TARGET_CITY, num_routes, num_buses, bus_stop_json
    )
    print("\n노선 생성이 완료되었습니다!")
    print(f"지도는 'bus_routes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html' 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()

# compute_fitness 함수 (간단한 적합도 계산 예시)
def compute_fitness(route, meta):
    stop_ids = [s["id"] for s in route]
    stops = [(s["lat"], s["lon"]) for s in route]

    # F1: 노선 길이
    D_list = [geodesic(stops[i], stops[i+1]).meters for i in range(len(stops)-1)]
    L = sum(D_list)
    F1 = route_length(L, L_i=10000, L_max=15000)

    # F2: 정류장 간 거리
    F2 = stop_distance(D_list, D_ideal=400, sigma=200)

    # F3: POI 커버리지
    poi_list = meta["poi_lookup_func"](route)
    F3 = poi_score(poi_list, meta["poi_weights"])

    # F4: 지하철 접근성
    subway_dists = meta["subway_distance_func"](route)
    F4 = subway_distance(subway_dists, D_scale=1000)

    # F6: 지역 공정성
    weights, counts = meta["duplication_func"](route)
    F6 = duplication_penalty_score(weights, counts)

    # F7: 환승 정류장 포함 비율
    F7 = compute_transfer_score(stop_ids, meta["transfer_stop_ids"])

    # F8: 링크 중심성 (정규화 점수 0~1)
    F8 = meta["link_centrality_func"](route)

    # F9: 노드 중심성 (정규화 점수 0~1)
    F9 = meta["node_centrality_func"](route)

    # F10: 정류장 수 적정성
    F10 = normalize_stop_count(len(route), N_ideal=30, N_max=15)

    # 가중치 적용
    W = meta["weights"]
    fitness = (
        W["w1"] * F1 + W["w2"] * F2 + W["w3"] * F3 + W["w4"] * F4 +
        W["w6"] * F6 + W["w7"] * F7 + W["w8"] * F8 + W["w9"] * F9 + W["w10"] * F10
    )
    return fitness
