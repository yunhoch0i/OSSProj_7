import os
import pandas as pd
import requests
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# .env 파일에서 환경 변수 읽어오기
load_dotenv()

# 1) API 키 관리
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")
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
    r"C:\Users\지동우\Desktop\공소 자료\project\정류장 위치\과천시_버스_정류장_위치.xlsx",
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
CATEGORIES = {
    "대형마트": "MT1",
    "편의점": "CS2",
    "어린이집": "PS3",
    "학교": "SC4",
    "학원": "AC5",
    "주차장": "PK6",
    "주유소": "OL7",
    "지하철역": "SW8",
    "은행": "BK9",
    "문화시설": "CT1",
    "중개업소": "AG2",
    "공공기관": "PO3",
    "관광명소": "AT4",
    "숙박": "AD5",
    "음식점": "FD6",
    "카페": "CE7",
    "병원": "HP8",
    "약국": "PM9"
}

# 5) 정류장별 POI 수집
stops = df.to_dict(orient="records")
poi_data = []

for stop in stops:
    lon, lat = stop["lon"], stop["lat"]
    poi_counts = {}
    for label, code in CATEGORIES.items():
        count = search_category_nearby(lon, lat, code, radius=300)
        poi_counts[label] = count
    
    # POI 데이터를 리스트에 추가
    poi_row = {
        "정류소ID": stop["id"],
        "정류소명": stop["name"],
        "위도": stop["lat"],
        "경도": stop["lon"]
    }
    # POI 카운트 데이터 추가
    for label, count in poi_counts.items():
        poi_row[f"POI_{label}"] = count
    
    poi_data.append(poi_row)

# 6) DataFrame 생성 및 CSV 저장
poi_df = pd.DataFrame(poi_data)
output_file = "bus_stop_poi.csv"
poi_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"POI 정보가 {output_file}에 저장되었습니다.")


