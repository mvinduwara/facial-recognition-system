package com.facerec;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.kafka.annotation.EnableKafka;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableKafka
@EnableAsync
@EnableScheduling
public class FacialRecognitionSystemApplication {
    public static void main(String[] args) {
        // Line removed! JavaCV will auto-load the native libraries for us.
        SpringApplication.run(FacialRecognitionSystemApplication.class, args);
    }
}