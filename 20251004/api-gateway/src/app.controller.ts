// 필요한 부품들을 @nestjs 패키지에서 가져옵니다.
import { Controller, Post, Body, Inject, OnModuleInit, Logger } from '@nestjs/common';
import { ClientKafka } from '@nestjs/microservices';

// 재시도를 위한 헬퍼 함수입니다. ms 밀리초만큼 잠시 기다리는 역할을 합니다.
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// @Controller 데코레이터는 이 클래스가 외부 요청을 처리하는 컨트롤러임을 선언합니다.
@Controller()
// OnModuleInit 인터페이스는 모듈이 초기화된 후 특정 작업을 수행하고 싶을 때 사용합니다.
export class AppController implements OnModuleInit {
  // 로그를 예쁘게 찍기 위해 NestJS의 기본 로거를 생성합니다.
  private readonly logger = new Logger(AppController.name);

  // 생성자(constructor)를 통해 의존성을 주입받습니다.
  constructor(
    // @Inject('KAFKA_SERVICE')는 AppModule에서 등록한 Kafka 클라이언트를
    // 'kafkaClient'라는 클래스 멤버 변수에 주입해달라고 NestJS에 요청하는 것입니다.
    @Inject('KAFKA_SERVICE') private readonly kafkaClient: ClientKafka,
  ) {}

  // onModuleInit: 이 서비스(모듈)가 시작될 때 딱 한 번 실행되는 함수입니다.
  async onModuleInit() {
    // Kafka 연결을 위한 재시도(Retry) 로직입니다.
    const maxRetries = 5; // 최대 5번까지 재시도
    let retries = 0;
    while (retries < maxRetries) {
      try {
        await this.kafkaClient.connect(); // Kafka 서버에 접속을 시도합니다.
        this.logger.log('API Gateway: Kafka client connected successfully.');
        return; // 접속에 성공하면 함수를 종료합니다.
      } catch (err) {
        retries++;
        this.logger.error(
          `Failed to connect to Kafka. Retrying... (${retries}/${maxRetries})`,
          err.stack,
        );
        await sleep(2000); // 2초 기다렸다가 다시 시도합니다.
      }
    }
    // 최대 재시도 횟수를 초과하면, 심각한 문제이므로 로그를 남기고 서비스를 종료합니다.
    this.logger.error('Could not connect to Kafka after multiple retries. Exiting.');
    process.exit(1);
  }

  // @Post('/request'): HTTP POST 방식으로 '/request' 경로에 요청이 들어오면 이 함수를 실행합니다.
  @Post('/request')
  // @Body() 데코레이터는 요청의 본문(body)에 담긴 JSON 데이터를 'data' 변수에 담아줍니다.
  handleUserRequest(@Body() data: any) {
    this.logger.log(`API Gateway received request: ${JSON.stringify(data)}`);
    // 'kafkaClient'를 사용해 'user-requests'라는 토픽(주제)으로 데이터를 메시지로 보냅니다.
    this.kafkaClient.emit('user-requests', JSON.stringify(data));
    // 사용자에게는 즉시 "요청이 접수되었습니다"라고 응답하여, 기다리지 않게 합니다.
    return { message: 'Your request has been accepted and is being processed.' };
  }
}