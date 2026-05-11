package com.facerec.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import java.util.concurrent.Executor;

@Configuration
public class AsyncConfig {

    @Bean(name = "frameProcessorExecutor")
    public Executor frameProcessorExecutor() {
        ThreadPoolTaskExecutor exec = new ThreadPoolTaskExecutor();
        exec.setCorePoolSize(4); // 4 concurrent frame processors
        exec.setMaxPoolSize(8);  // Burst up to 8
        exec.setQueueCapacity(50); // Buffer 50 frames max
        exec.setThreadNamePrefix("frame-proc-");

        // Drop frames if overloaded (prefer freshness)
        exec.setRejectedExecutionHandler((r, executor) -> {});
        exec.initialize();
        return exec;
    }
}