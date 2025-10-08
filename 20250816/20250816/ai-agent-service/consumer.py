import json
from kafka import KafkaConsumer # kafka-python 라이브러리에서 KafkaConsumer 클래스를 가져옵니다.
from kafka.errors import NoBrokersAvailable # Kafka 서버를 찾지 못했을 때 발생하는 에러
import time
from agents.graph_builder import create_agent_workflow
from agents.graph_state import AgentState

# Kafka에 접속을 시도하는 함수입니다.
def create_consumer(retries=5, delay=5):
    # 최대 5번까지 재시도를 합니다.
    for i in range(retries):
        try:
            # Kafka 컨슈머(메시지를 받는 역할)를 생성하고 접속을 시도합니다.
            consumer = KafkaConsumer(
                'user-requests', # "저는 'user-requests' 토픽(채팅방)을 구독하겠습니다."
                bootstrap_servers='kafka:29092', # 접속할 Kafka 서버 주소 ('kafka'는 docker-compose 서비스 이름)
                auto_offset_reset='earliest', # 만약 제가 처음 접속한다면, 가장 오래된 메시지부터 읽겠습니다.
                enable_auto_commit=True, # 메시지를 읽으면 자동으로 '읽음' 처리합니다.
                group_id='ai-agent-group', # 저희 팀 이름은 'ai-agent-group'입니다. (같은 팀은 메시지를 나눠서 처리)
                # Kafka로부터 받은 메시지(바이트)를 UTF-8로 디코딩한 후 JSON으로 변환해달라는 설정입니다.
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            print("AI Agent Service: Kafka Consumer connected successfully.")
            return consumer # 접속에 성공하면 생성된 컨슈머 객체를 반환합니다.
        except NoBrokersAvailable:
            # Kafka 서버를 찾지 못하면, 잠시 기다렸다가 다시 시도합니다.
            print(f"Could not connect to Kafka. Retrying in {delay} seconds... ({i+1}/{retries})")
            time.sleep(delay)
    print("Could not connect to Kafka after multiple retries. Exiting.")
    return None # 계속 실패하면 None을 반환합니다.


if __name__ == "__main__":
    print("AI Agent Service is starting...")
    consumer = create_consumer()
    
    # AI 조직(워크플로우)을 컴파일합니다.
    app = create_agent_workflow()

    if consumer:
        print("Waiting for messages...")
        for message in consumer:
            request_data = message.value
            print(f"\n--- New Project (ID: {request_data.get('request_id')}) Started ---")
            
            # [핵심 수정] AgentState 정의에 맞게 모든 데이터를 완벽하게 기입합니다.
            initial_state: AgentState = {
                "project_id": request_data.get("request_id"),
                "user_request": request_data.get("user_request"),
                "retrieved_data_summary": request_data.get("retrieved_data_summary", {}),
                "analysis_options": request_data.get("analysis_options", {}),
                "product_type": request_data.get("product_type"), # product_type도 추가
                
                # 나머지 필드들은 워크플로우가 진행되면서 채워질 것이므로 None으로 초기화합니다.
                "cx_insights": None,
                "topics": None,
                "personas": None,
                "service_ideas": None,
                "data_plan": None,
                "final_document": None,
                "current_observation": "Project started from Kafka message.",
                "next_action": None
            }
            
            # 완성된 서류(initial_state)를 가지고 AI 조직의 워크플로우를 시작합니다.
            final_state = app.invoke(initial_state)
            
            print(f"--- Project Finished ---")
            print(f"Final Result: {json.dumps(final_state, indent=2, ensure_ascii=False)}")