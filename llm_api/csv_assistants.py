import openai
import time
import re
from typing import List, Dict
import os

# location_name = "연천군"
# location_name = "고양시"
location_name = "용인시"

class GPTAssistantAnalyzer:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.assistant_id = None
        self.thread_id = None
        
    def create_assistant(self):
        """CSV 분석용 GPT Assistant 생성"""
        assistant = self.client.beta.assistants.create(
            name="CSV Transportation Analyzer",
            instructions="""당신은 교통 데이터 분석 전문가입니다. 
            제공된 CSV 파일들을 종합적으로 분석하여 버스 노선 설계에 필요한 
            10개의 가중치를 정량적으로 산정해주세요.""",
            model="gpt-4-turbo",
            tools=[{"type": "code_interpreter"}]
        )
        self.assistant_id = assistant.id
        return assistant.id
    
    def upload_files(self, csv_files: List[str]) -> List[str]:
        """CSV 파일들을 Assistant에 업로드"""
        file_ids = []
        for file_path in csv_files:
            try:
                with open(file_path, 'rb') as f:
                    file = self.client.files.create(
                        file=f,
                        purpose='assistants'
                    )
                    file_ids.append(file.id)
                    print(f"업로드 완료: {file_path} -> {file.id}")
            except Exception as e:
                print(f"파일 업로드 실패 {file_path}: {e}")
        return file_ids
    
    def create_thread_and_analyze(self, file_ids: List[str], prompt: str):
        """스레드 생성 및 분석 실행"""
        # 스레드 생성
        thread = self.client.beta.threads.create()
        self.thread_id = thread.id
        
        # 파일들을 스레드에 첨부하여 메시지 생성
        message = self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=prompt,
            attachments=[{"file_id": fid, "tools": [{"type": "code_interpreter"}]} 
                        for fid in file_ids]
        )
        
        # Assistant 실행
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id
        )
        
        return self.wait_for_completion(run.id)
    
    def wait_for_completion(self, run_id: str, max_wait: int = 300):
        """실행 완료 대기"""
        start_time = time.time()
        while time.time() - start_time < max_wait:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id,
                run_id=run_id
            )
            
            if run.status == "completed":
                messages = self.client.beta.threads.messages.list(
                    thread_id=self.thread_id
                )
                return messages.data[0].content[0].text.value
            
            elif run.status in ["failed", "cancelled", "expired"]:
                raise Exception(f"실행 실패: {run.status}")
            
            time.sleep(5)
            print(f"분석 진행 중... 상태: {run.status}")
        
        raise TimeoutError("분석 시간 초과")
    
    def normalize_key(self, key: str) -> str:
        """키 정규화"""
        key_mapping = {
            '노선길이': '노선 길이',
            '정류장간거리': '정류장 간 거리', 
            'poi정보': 'POI 정보',
            '지하철역거리': '지하철 역과의 거리',
            '유동인구': '유동인구',
            '지역공정성': '지역 공정성',
            '환승여부': '환승 여부',
            '교통링크': '교통 링크',
            '교통노드': '교통 노드',
            '정류장수': '정류장 수'
        }
        
        key_lower = key.replace(' ', '').lower()
        for k, v in key_mapping.items():
            if k.replace(' ', '').lower() in key_lower:
                return v
        return key

    
    def cleanup(self):
        """리소스 정리"""
        if self.assistant_id:
            self.client.beta.assistants.delete(self.assistant_id)
            print("Assistant 삭제 완료")

def main():
    # 설정
    csv_files = [
        rf"reference/{location_name}_POI.csv",
        rf"reference/{location_name}_교통노드.csv", 
        rf"reference/{location_name}_교통링크.csv",
        rf"reference/{location_name}_위치.csv"
    ]

    prompt = f"""다음 조건에 따라, 성공적인 버스 정책을 실행한 {location_name}의 교통 데이터를 기반으로 버스 노선 설계에 필요한 10개의 가중치를 정량적으로 산정해주세요. csv파일만을 대상으로 분석하면 됩니다. 다음의 조건외에 추가적인 지시가 필요한 부분은 알아서 판단해주세요.
추가적으로 사용자에게 질문하지 않고 가중치 계산해주세요.
- 이 분석은 {location_name} 데이터를 통해 추출된 성공적인 정책의 패턴을 일반화하고, 향후 과천시에 해당 정책을 적용할 때 활용될 수 있는 적합도 함수의 가중치를 정의하기 위한 목적입니다.
- 과천시 데이터는 제공되지 않으며, 오직 {location_name} 데이터를 기반으로 분석이 이루어져야 합니다.
- 추출된 가중치는 향후 과천시 노선도 재설계 시 적합도 평가 함수 또는 최적화 모델에 직접 사용될 예정입니다.
- 가중치는 다음의 조건을 만족해야 합니다:
  - 각각의 가중치는 0과 1 사이의 실수입니다.
  - 모든 가중치의 합은 정확히 1이 되어야 합니다.
  - 예를 들어, 노선의 길이가 상대적으로 더 중요하다면, 해당 항목의 가중치는 다른 항목보다 더 크게 설정되어야 하며, 그에 따라 다른 항목의 가중치는 비례적으로 조정되어야 합니다.
  - 최종 답변을 출력하기 전에 가중치의 합이 1이 되는지 확인하고, 그렇지 않다면 다시 계산해야 합니다.

분석 대상 요소는 다음과 같습니다:
w1. 노선 길이
w2. 정류장 간 거리
w3. POI(Point of Interest) 정보
w4. 지하철 역과의 거리
w5. 유동인구
w6. 지역 공정성 (소외 지역 접근성 등)
w7. 환승 여부
w8. 교통 링크 (도로망 연결성)
w9. 교통 노드 (중심성, 허브 여부)
w10. 정류장 수

각 항목에 대해 {location_name} 데이터 내의 유의미한 패턴을 추출하고, 항목별 중요도를 수치로 반영하여 아래 형식에 따라 출력해주세요. 전체 답변의 마지막에는 아래의 형식에 따른 가중치 정보들만 출력하고 다른 답변없이 종료하면 됩니다:

예시 출력 형식:
w1: 0.154  
w2: 0.090  
w3: 0.085  
w4: 0.120  
… (이하 생략)"""
    
    api_key = ""
    
    analyzer = GPTAssistantAnalyzer(api_key)
    
    try:
        # Assistant 생성
        print("GPT Assistant 생성 중...")
        analyzer.create_assistant()
        
        # 파일 업로드
        print("CSV 파일 업로드 중...")
        file_ids = analyzer.upload_files(csv_files)
        
        if not file_ids:
            raise Exception("업로드된 파일이 없습니다.")
        
        # 분석 실행
        print("데이터 분석 시작...")
        response = analyzer.create_thread_and_analyze(file_ids, prompt)
        
        print("=== GPT Assistant 응답 ===")
        print(response)
        print("=" * 50)
        
        # 가중치 추출
        # weights = analyzer.extract_weights(response)
        
        with open("llm_response.txt", "w", encoding="utf-8") as f:
            f.write(response)
            print("LLM 응답이 'llm_response.txt' 파일로 저장되었습니다.")
            
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        analyzer.cleanup()

if __name__ == "__main__":
    main()