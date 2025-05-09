import networkx as nx
import osmnx as ox
from math import radians, sin, cos, sqrt, atan2
import random
import numpy as np
from sklearn.cluster import KMeans
import folium
from heapq import heappush, heappop
from collections import Counter
import csv
import io
from typing import Dict, List, Tuple, Set, Any

# 상수 정의
MIN_ROAD_WIDTH = 6
TARGET_CITY = "Gwacheon"
OSM_FILE_PATH = "./map.osm"  # map.osm 파일 경로

# Haversine 공식으로 거리 계산 (단위: km)
def haversine(lon1, lat1, lon2, lat2):
    """두 지점 간의 거리를 계산합니다 (단위: km)"""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# A* 알고리즘 구현
def astar_path(G, start, goal, constraints=None):
    """A* 알고리즘을 사용하여 두 노드 간의 최단 경로를 찾습니다."""
    if constraints is None:
        constraints = {"min_width": MIN_ROAD_WIDTH}

    def heuristic(node1, node2):
        lon1, lat1 = G.nodes[node1]["pos"]
        lon2, lat2 = G.nodes[node2]["pos"]
        return haversine(lon1, lat1, lon2, lat2)

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current_f, current = heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor in G.neighbors(current):
            if "width" in G[current][neighbor] and G[current][neighbor]["width"] < constraints.get("min_width", 0):
                continue

            tentative_g_score = g_score[current] + G[current][neighbor]["weight"]

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heappush(open_set, (f_score[neighbor], neighbor))

    return None

# CSV에서 정류장 데이터 로드 및 OSM 정류장 병합
def load_bus_stops(csv_data, osm_stops):
    """bus_stop.csv와 OSM 정류장 데이터를 병합합니다."""
    stops = []
    csv_reader = csv.DictReader(io.StringIO(csv_data))

    # CSV 정류장 처리
    csv_stops = {}
    for i, row in enumerate(csv_reader):
        name = row["정류장명"].strip()
        if not name:
            print(f"경고: 행 {i+2} - 정류장명이 비어 있습니다. 스킵합니다.")
            continue
        try:
            lon = float(row["경도(WGS84)"])
            lat = float(row["위도(WGS84)"])
            stop_id = f"csv_stop_{name}_{i}"
            csv_stops[stop_id] = {
                "id": stop_id,
                "name": name,
                "lat": lat,
                "lon": lon,
                "passengers": 100,
                "population": 500
            }
        except (ValueError, KeyError) as e:
            print(f"경고: 행 {i+2} - 데이터 오류 ({e}). 스킵합니다.")

    # OSM 정류장과 병합 (거리 기준으로 중복 제거)
    for osm_stop in osm_stops:
        closest_csv = min(csv_stops.items(),
                          key=lambda x: haversine(osm_stop["lon"], osm_stop["lat"], x[1]["lon"], x[1]["lat"]),
                          default=(None, None))[1]
        if closest_csv and haversine(osm_stop["lon"], osm_stop["lat"], closest_csv["lon"], closest_csv["lat"]) < 0.05:
            stops.append(closest_csv)
        else:
            stops.append(osm_stop)

    print(f"병합된 정류장 수: {len(stops)}")
    return stops

# OSM 데이터 가져오기
def fetch_osm_data(city_name, osm_file=OSM_FILE_PATH):
    """map.osm 파일에서 도로, 정류장, POI 데이터를 가져옵니다."""
    print(f"{city_name}의 OSM 데이터를 {osm_file}에서 가져오는 중...")

    # 도로 네트워크 로드
    G_road = ox.graph_from_xml(osm_file)

    # 차량 통행 가능한 도로만 필터링
    drive_highways = [
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "motorway_link", "trunk_link",
        "primary_link", "secondary_link", "tertiary_link"
    ]
    edges_to_remove = [
        (u, v, k) for u, v, k, data in G_road.edges(keys=True, data=True)
        if data.get("highway") not in drive_highways
    ]
    G_road.remove_edges_from(edges_to_remove)

    # 정류장 데이터 로드
    tags = {"highway": "bus_stop"}
    gdf_stops = ox.features_from_xml(osm_file, tags=tags, filter_types=["node"])

    # POI 데이터 로드
    poi_tags = {"amenity": True, "shop": True, "leisure": True}
    gdf_pois = ox.features_from_xml(osm_file, tags=poi_tags, filter_types=["node"])

    # 정류장 데이터 가공
    stops = []
    for idx, row in gdf_stops.iterrows():
        if row.geometry.type == "Point":
            stops.append({
                "id": f"osm_stop_{idx}",
                "name": row.get("name", "Unnamed Stop"),
                "lat": row.geometry.y,
                "lon": row.geometry.x,
                "passengers": 100,
                "population": 500
            })

    # POI 데이터 가공
    pois = []
    for idx, row in gdf_pois.iterrows():
        if row.geometry.type == "Point":
            poi_type = row.get("amenity", row.get("shop", row.get("leisure", "unknown")))
            pois.append({
                "id": f"osm_poi_{idx}",
                "type": poi_type,
                "lat": row.geometry.y,
                "lon": row.geometry.x
            })

    # 도로 데이터 가공
    roads = []
    for u, v, data in G_road.edges(data=True):
        roads.append({
            "id": f"road_{u}_{v}",
            "start": {"lat": G_road.nodes[u]["y"], "lon": G_road.nodes[u]["x"]},
            "end": {"lat": G_road.nodes[v]["y"], "lon": G_road.nodes[v]["x"]},
            "length": data.get("length", 0) / 1000,
            "width": 6
        })

    return {
        "pois": pois,
        "roads": roads,
        "stops": stops,
        "road_graph": G_road
    }

# 노드 레이블링 (K-means 클러스터링)
def label_nodes(pois, stops, k=3):
    """POI 데이터를 기반으로 정류장에 레이블을 할당합니다."""
    if len(pois) < k:
        k = max(2, len(pois))

    poi_coords = np.array([[poi["lat"], poi["lon"]] for poi in pois])
    poi_types = [poi["type"] for poi in pois]

    kmeans = KMeans(n_clusters=k, random_state=0).fit(poi_coords)

    cluster_types = {}
    for i in range(k):
        cluster_pois = [poi_types[j] for j in range(len(poi_types)) if kmeans.labels_[j] == i]
        most_common = Counter(cluster_pois).most_common(1)
        cluster_types[i] = most_common[0][0] if most_common else "기타"

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

    stop_labels = {}
    for stop in stops:
        nearby_pois = [i for i, poi in enumerate(pois)
                       if haversine(stop["lon"], stop["lat"], poi["lon"], poi["lat"]) <= 0.3]
        if nearby_pois:
            cluster = Counter(kmeans.labels_[nearby_pois]).most_common(1)[0][0]
            stop_labels[stop["id"]] = cluster_types[cluster]
        else:
            stop_labels[stop["id"]] = "기타"

    return stop_labels

# 그래프 생성 함수
def create_city_graph(stops, pois, roads, osm_road_graph):
    """OSM 도로 그래프와 정류장 데이터를 결합하여 네트워크 그래프를 생성합니다."""
    G = nx.Graph()
    stop_labels = label_nodes(pois, stops)

    # 정류장 노드 추가
    for stop in stops:
        G.add_node(stop["id"],
                   pos=(stop["lon"], stop["lat"]),
                   name=stop["name"],
                   label=stop_labels[stop["id"]],
                   passengers=stop["passengers"],
                   population=stop["population"])

    # OSM 도로 그래프에서 엣지 추가
    for u, v, data in osm_road_graph.edges(data=True):
        length = data.get("length", 0) / 1000
        G.add_edge(f"node_{u}", f"node_{v}", weight=length, width=6, highway=data.get("highway", ""))

    # 정류장을 도로 노드에 연결
    for stop in stops:
        nearest_node = min(osm_road_graph.nodes,
                           key=lambda n: haversine(stop["lon"], stop["lat"],
                                                   osm_road_graph.nodes[n]["x"], osm_road_graph.nodes[n]["y"]))
        distance = haversine(stop["lon"], stop["lat"],
                             osm_road_graph.nodes[nearest_node]["x"], osm_road_graph.nodes[nearest_node]["y"])
        G.add_edge(stop["id"], f"node_{nearest_node}", weight=distance, width=6)

    # 그래프 연결성 디버깅
    print(f"그래프 노드 수: {G.number_of_nodes()}, 엣지 수: {G.number_of_edges()}")
    isolated = list(nx.isolates(G))
    if isolated:
        print(f"연결되지 않은 노드: {isolated}")

    return G

# 유전 알고리즘 적합도 함수
def fitness(route, G, reference_weights):
    """노선의 적합도를 계산합니다."""
    if len(route) < 2:
        return 0

    length = 0
    for i in range(len(route) - 1):
        path = astar_path(G, route[i], route[i + 1])
        if path and len(path) > 1:
            path_length = sum(G[path[j]][path[j + 1]]["weight"] for j in range(len(path) - 1))
            length += path_length
        else:
            return 0

    stop_count = len(route)
    area_types = Counter([G.nodes[stop]["label"] for stop in route])
    passengers = sum(G.nodes[stop]["passengers"] for stop in route)
    avg_passengers = passengers / stop_count
    avg_distance = length / (stop_count - 1) if stop_count > 1 else 0

    # 도로 유형 점수 (주요 도로 선호)
    road_type_score = sum(1 if G[u][v].get("highway", "") in ["primary", "secondary"] else 0.5
                          for u, v in zip(route[:-1], route[1:]) if G.has_edge(u, v))

    length_diff = abs(length - reference_weights["avg_length"]) / reference_weights["avg_length"]
    stop_count_diff = abs(stop_count - reference_weights["avg_stop_count"]) / reference_weights["avg_stop_count"]
    passengers_diff = abs(avg_passengers - reference_weights["avg_passengers"]) / reference_weights["avg_passengers"]
    distance_diff = abs(avg_distance - reference_weights["avg_distance"]) / reference_weights["avg_distance"]

    area_type_diff = 0
    for area_type, ref_ratio in reference_weights["area_type_distribution"].items():
        route_ratio = area_types[area_type] / stop_count if stop_count > 0 else 0
        area_type_diff += abs(route_ratio - ref_ratio)

    fitness_score = 10 - (
            0.3 * length_diff +
            0.2 * stop_count_diff +
            0.25 * passengers_diff +
            0.15 * distance_diff +
            0.1 * area_type_diff
    ) + 0.1 * road_type_score

    return max(0, fitness_score)

# 유전 알고리즘
def genetic_algorithm(G, reference_weights, num_routes, population_size=50, generations=100):
    """유전 알고리즘을 사용하여 버스 노선을 생성합니다."""
    all_stops = [n for n in G.nodes() if "stop" in n]
    if len(all_stops) < 3:
        print("에러: 그래프에 정류장이 충분하지 않습니다.")
        return []

    hub = max(all_stops, key=lambda x: G.nodes[x]["passengers"])
    target_stop_count = int(reference_weights["avg_stop_count"])

    # 초기 인구 생성
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

    for gen in range(generations):
        scored_population = [(route, fitness(route, G, reference_weights)) for route in population]
        scored_population.sort(key=lambda x: x[1], reverse=True)
        new_population = [route for route, _ in scored_population[:population_size//2]]

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(scored_population[:population_size//2], 2)
            child = parent1[0].copy()
            if len(parent1[0]) > 2 and len(parent2[0]) > 2:
                crossover_point = random.randint(1, min(len(parent1[0]), len(parent2[0]))-1)
                child = parent1[0][:crossover_point]
                child.extend([s for s in parent2[0] if s not in child])

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

    final_scored_population = [(route, fitness(route, G, reference_weights)) for route in population]
    final_scored_population.sort(key=lambda x: x[1], reverse=True)
    return [route for route, _ in final_scored_population[:num_routes]]

# 버스 스케줄링
def schedule_buses(routes, G, num_buses):
    """버스 노선을 버스에 할당하고 스케줄을 생성합니다."""
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
        round_trip_time = (length * 2) / 30 * 60
        route_stats.append({
            "route_id": i,
            "route": route,
            "length": length,
            "passengers": passengers,
            "round_trip_time": round_trip_time
        })

    route_stats.sort(key=lambda x: x["round_trip_time"], reverse=True)
    schedule = {f"Bus {i+1}": [] for i in range(num_buses)}
    for route_info in route_stats:
        bus_times = {bus: sum(r["round_trip_time"] for r in schedule[bus]) for bus in schedule}
        target_bus = min(bus_times, key=bus_times.get)
        schedule[target_bus].append(route_info)
    return schedule

# 배차 간격 계산
def calculate_frequencies(schedule):
    """각 노선의 배차 간격을 계산합니다."""
    frequencies = {}
    for bus, routes in schedule.items():
        for route_info in routes:
            route_id = route_info["route_id"]
            bus_count = sum(1 for b, rs in schedule.items() for r in rs if r["route_id"] == route_id)
            frequency = max(5, min(60, route_info["round_trip_time"] / bus_count))
            frequencies[route_id] = round(frequency)
    return frequencies

# 노선 특성 계산
def calculate_route_characteristics(routes, G):
    """생성된 노선들의 특성을 계산합니다."""
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

# Folium 지도 시각화
def display_routes(G, routes, schedule, frequencies):
    """버스 노선을 지도에 시각화합니다."""
    pos = nx.get_node_attributes(G, "pos")
    lats = [G.nodes[node]["pos"][1] for node in G.nodes() if "stop" in node]
    lons = [G.nodes[node]["pos"][0] for node in G.nodes() if "stop" in node]
    center_lat = sum(lats) / len(lats) if lats else 37.43
    center_lon = sum(lons) / len(lons) if lons else 126.99
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

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
        popup_text = f"{name} ({label})<br>승객: {passengers}/일"
        folium.Marker([lat, lon], popup=popup_text, icon=folium.Icon(color=icon_color)).add_to(m)

    route_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple']
    for i, route in enumerate(routes):
        route_id = i
        route_color = route_colors[i % len(route_colors)]
        frequency = frequencies.get(route_id, 0)
        for j in range(len(route) - 1):
            start = route[j]
            end = route[j + 1]
            path = astar_path(G, start, end)
            if path and len(path) > 1:
                path_coords = [(G.nodes[node]["pos"][1], G.nodes[node]["pos"][0]) for node in path]
                folium.PolyLine(
                    path_coords,
                    color=route_color,
                    weight=4,
                    opacity=0.8,
                    popup=f"노선 {route_id+1} (배차간격: {frequency}분)"
                ).add_to(m)

    for bus, routes_info in schedule.items():
        for route_info in routes_info:
            route_id = route_info["route_id"]
            route = route_info["route"]
            if route:
                start_node = route[0]
                lat, lon = G.nodes[start_node]["pos"][1], G.nodes[start_node]["pos"][0]
                popup_text = f"{bus}<br>노선 {route_id+1}<br>배차간격: {frequencies.get(route_id, 0)}분"
                folium.Marker(
                    [lat, lon],
                    popup=popup_text,
                    icon=folium.Icon(icon="bus", prefix="fa", color="white")
                ).add_to(m)

    m.save("bus_routes.html")
    print("지도는 'bus_routes.html' 파일로 저장되었습니다.")
    return m

# 결과 요약 및 출력
def print_route_summary(routes, G, schedule, frequencies, reference_weights=None):
    """생성된 노선의 요약 정보를 출력합니다."""
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
            avg_distance = length / (len(route) - 1)
            generated_avg_distance.append(avg_distance)
            route_passengers = sum(G.nodes[stop]["passengers"] for stop in route) / len(route)
            generated_passengers.append(route_passengers)
            for stop in route:
                area_type_counts[G.nodes[stop]["label"]] += 1

        if generated_lengths:
            print(f"노선 평균 길이: {np.mean(generated_lengths):.2f} km (참조: {reference_weights['avg_length']:.2f} km)")
            print(f"노선 평균 정류장 수: {np.mean(generated_stop_counts):.2f} (참조: {reference_weights['avg_stop_count']:.2f})")
            print(f"정류장 간 평균 거리: {np.mean(generated_avg_distance):.2f} km (참조: {reference_weights['avg_distance']:.2f} km)")
            print(f"정류장 평균 이용객 수: {np.mean(generated_passengers):.2f}/일 (참조: {reference_weights['avg_passengers']:.2f}/일)")
            print("\n지역 유형 분포:")
            total_stops = sum(area_type_counts.values())
            for area_type, count in area_type_counts.most_common():
                generated_ratio = count / total_stops
                reference_ratio = reference_weights["area_type_distribution"].get(area_type, 0)
                print(f"  - {area_type}: {generated_ratio:.2%} (참조: {reference_ratio:.2%})")

# 메인 함수
def generate_bus_routes(reference_weights, target_city, num_routes, num_buses, bus_stop_csv):
    """OSM 데이터를 사용하여 버스 노선을 생성합니다."""
    print(f"\n1. {target_city}의 교통 네트워크 구축 중...")
    osm_data = fetch_osm_data(target_city)
    stops = load_bus_stops(bus_stop_csv, osm_data["stops"])
    target_graph = create_city_graph(stops, osm_data["pois"], osm_data["roads"], osm_data["road_graph"])

    print(f"\n2. {target_city}에 최적 버스 노선 생성 중...")
    routes = genetic_algorithm(target_graph, reference_weights, num_routes)

    print(f"\n3. 버스 배차 계획 수립 중...")
    schedule = schedule_buses(routes, target_graph, num_buses)
    frequencies = calculate_frequencies(schedule)

    print_route_summary(routes, target_graph, schedule, frequencies, reference_weights)
    print(f"\n4. 노선도 생성 중...")
    map_object = display_routes(target_graph, routes, schedule, frequencies)

    return routes, schedule, frequencies, map_object

# 명령줄 인터페이스
if __name__ == "__main__":
    print("===== 버스 노선 생성 시스템 =====")
    print(f"대상 도시: {TARGET_CITY}")

    reference_weights = {
        "avg_length": 5.0,
        "avg_stop_count": 12.0,
        "avg_passengers": 200.0,
        "avg_distance": 0.7,
        "area_type_distribution": Counter({
            "상업지구": 0.4,
            "주거지역": 0.3,
            "교육지구": 0.2,
            "기타": 0.1
        })
    }

    with open("bus_stop.csv", "r", encoding="utf-8") as f:
        bus_stop_csv = f.read()

    num_routes = int(input("생성할 노선 개수를 입력하세요: "))
    num_buses = int(input("운행 가능한 버스 대수를 입력하세요: "))

    routes, schedule, frequencies, map_object = generate_bus_routes(
        reference_weights, TARGET_CITY, num_routes, num_buses, bus_stop_csv
    )

    print("\n노선 생성이 완료되었습니다!")
    print("지도는 'bus_routes.html' 파일로 저장되었습니다.")