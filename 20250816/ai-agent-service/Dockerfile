# 1. 베이스 이미지: Python 3.9의 가벼운 slim 버전을 사용합니다.
FROM python:3.9-slim
# 2. 작업 디렉토리: 컨테이너 내부에서 작업할 기본 폴더를 설정합니다.
WORKDIR /app
# 3. 의존성 설치: 먼저 requirements.txt 파일을 복사하여 라이브러리를 설치합니다.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# 4. 소스코드 복사: 나머지 모든 프로젝트 파일을 컨테이너로 복사합니다.
COPY . .
# 5. 실행: 컨테이너가 시작될 때 consumer.py 파일을 실행합니다.
# '-u' 옵션은 unbuffered를 의미하며, 로그가 버퍼에 쌓이지 않고 즉시 출력되게 합니다.
CMD ["python", "-u", "consumer.py"]