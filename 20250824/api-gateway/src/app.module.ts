// NestJS와 Kafka 연동에 필요한 부품들을 @nestjs 패키지에서 가져옵니다.
import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { ClientsModule, Transport } from '@nestjs/microservices';

// @Module 데코레이터는 AppModule 클래스가 NestJS의 모듈임을 선언합니다.
// 모듈은 관련된 컨트롤러, 서비스 등을 묶어주는 역할을 합니다.
@Module({
  imports: [
    ClientsModule.register([
      {
        name: 'KAFKA_SERVICE',
        transport: Transport.KAFKA,
        options: {
          client: {
            brokers: ['kafka:29092'],
          },
          // [핵심 수정] 프로듀서(메시지 발행자)를 위한 재시도 옵션을 추가합니다.
          producer: {
            // 개발 편의성을 위해, 토픽이 없을 경우 자동으로 생성하도록 허용합니다.
            allowAutoTopicCreation: true,
            // 재시도 정책 설정
            retry: { 
              retries: 5, // 최대 5번까지 재시도
              initialRetryTime: 300 // 처음엔 300ms 기다렸다가 재시도
            }
          },
          consumer: {
            groupId: 'api-gateway-consumer',
          },
        },
      },
    ]),
  ],
  controllers: [AppController],
})
export class AppModule {}