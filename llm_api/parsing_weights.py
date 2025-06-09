import re

# txt 파일 읽기
with open("llm_response.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 정규표현식으로 w1~w10 가중치 추출
weights = dict(re.findall(r"(w[1-9]|w10):\s*([0-9]*\.?[0-9]+)", text))

# 문자열을 float으로 변환
weights = {k: float(v) for k, v in weights.items()}

# 숫자 기준으로 정렬하여 출력
for key in sorted(weights, key=lambda x: int(x[1:])):
    print(f"{key}: {weights[key]:.3f}")

total = sum(weights.values())
print(f"합계: {total:.3f}")