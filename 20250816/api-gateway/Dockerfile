# 1. 베이스 이미지: Node.js 18 버전의 가벼운 alpine 버전을 사용합니다.
FROM node:18-alpine

# 2. 작업 디렉토리: 컨테이너 내부에서 작업할 기본 폴더를 설정합니다.
WORKDIR /usr/src/app

# 3. 의존성 설치: package.json 파일을 먼저 복사하여 라이브러리를 설치합니다.
#    (소스코드가 바뀌지 않고 라이브러리만 그대로일 경우, Docker는 이 단계를 캐싱하여 빌드 속도를 높입니다.)
COPY package*.json ./
RUN npm install

# 4. 소스코드 복사: 나머지 모든 프로젝트 파일을 컨테이너로 복사합니다.
COPY . .

# 5. 빌드: TypeScript 코드를 NestJS가 실행할 수 있는 JavaScript 코드로 변환(빌드)합니다.
RUN npm run build

# 6. 실행: 컨테이너가 시작될 때 최종적으로 실행할 명령어를 정의합니다.
CMD ["node", "dist/main"]