import math
import numpy as np

def route_length(L, L_i, L_max):
    score = 1 - ((L - L_i) / L_max) ** 2
    return max(0, score)  # 점수가 0 미만이면 0으로 보정

def stop_distance(D_list, D_ideal, sigma):
    if len(D_list) < 2:
        return 0  # 계산 불가
    scores = [math.exp(-((d - D_ideal) ** 2) / (2 * sigma ** 2)) for d in D_list]
    return sum(scores) / (len(D_list))

def poi_score(poi_list, weight_dict):
    total_weighted = 0
    total_count = 0
    for poi in poi_list:
        weight = weight_dict.get(poi['type'], 0)
        total_weighted += weight * poi['count']
        total_count += 1
    return total_weighted / total_count if total_count > 0 else 0

def subway_distance(subway_dists, D_scale):
    if not subway_dists:
        return 0
    scores = [math.exp(-d / D_scale) for d in subway_dists]
    return sum(scores) / len(subway_dists)

# 5번 함수의 경우 유동인구를 조사하기 어려움이 있음