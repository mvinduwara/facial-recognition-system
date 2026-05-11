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
        nu.pattern.OpenCV.loadLocally();
        SpringApplication.run(FacialRecognitionSystemApplication.class, args);
    }
}