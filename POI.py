import os
import pandas as pd
import requests
from typing import List, Dict, Tuple


# 1) API 키 관리
KAKAO_REST_API_KEY = process.env.KAKAO_REST_API_KEY
HEADERS = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}

# 2) 카테고리별 POI 개수 조회 함수
def search_category_nearby(
        lon: float,
        lat: float,
        category_group_code: str,
        radius: int = 300
) -> int:
    """
    Kakao 카테고리 검색 API를 호출해
    해당 카테고리 코드 반경 내 POI 총개수를 반환.
    """
    url = "https://dapi.kakao.com/v2/local/search/category.json"
    params = {
        "x": lon,
        "y": lat,
        "radius": radius,
        "category_group_code": category_group_code,
        "size": 1  # 총 개수만 필요하므로 최소 크기
    }
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    meta = resp.json().get("meta", {})
    return meta.get("total_count", 0)

# 3) 정류장 엑셀 전처리
df = pd.read_excel(
    r"/Users/yunho/Downloads/정류장 위치/과천시_버스_정류장_위치.xlsx",
    header=1
)

# 필요한 컬럼만 선택하고 이름 정리
df = df.rename(columns={
    "정류소id": "id",
    "정류소명": "name",
    "WGS84위도": "lat",
    "WGS84경도": "lon"
})
df = df[["id", "name", "lat", "lon"]].dropna()

# 4) 탐색할 POI 카테고리 그룹 코드 목록
# FD6(음식점), CE7(카페), CS2(편의점), HP8(병원) 등
CATEGORIES = {
    "음식점": "FD6",
    "카페":   "CE7",
    "편의점": "CS2",
    "병원":   "HP8"
}

# 5) 정류장별 POI 수집
stops = df.to_dict(orient="records")
for stop in stops:
    lon, lat = stop["lon"], stop["lat"]
    poi_counts: Dict[str, int] = {}
    for label, code in CATEGORIES.items():
        count = search_category_nearby(lon, lat, code, radius=300)
        poi_counts[label] = count
    # 결과를 정류장 dict에 추가
    stop["poi_counts"] = poi_counts

# 6) 결과 확인
for stop in stops[:5]:
    print(f"{stop['name']} ({stop['id']}): {stop['poi_counts']}")

