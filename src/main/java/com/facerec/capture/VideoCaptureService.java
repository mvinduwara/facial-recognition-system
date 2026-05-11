package com.facerec.capture;

import com.facerec.detection.FaceDetector;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.awt.image.BufferedImage;
import java.util.concurrent.atomic.AtomicBoolean;

@Slf4j
@Service
@RequiredArgsConstructor
public class VideoCaptureService {

    private final FaceDetector faceDetector;
    private final FrameProcessor frameProcessor;
    private final AtomicBoolean running = new AtomicBoolean(false);

    @Async
    public void startCapture(String source) {
        try (FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(source)) {
            grabber.setImageWidth(1280);
            grabber.setImageHeight(720);
            grabber.setFrameRate(30);
            grabber.start();
            running.set(true);

            Java2DFrameConverter converter = new Java2DFrameConverter();
            log.info("Video capture started from: {}", source);

            while (running.get()) {
                Frame frame = grabber.grab();
                if (frame == null || frame.image == null) continue;

                BufferedImage bufferedImage = converter.getBufferedImage(frame);
                if (bufferedImage != null) {
                    frameProcessor.processFrame(bufferedImage);
                }
            }
            grabber.stop();
        } catch (Exception e) {
            log.error("Video capture error: {}", e.getMessage(), e);
        }
    }

    public void stopCapture() {
        running.set(false);
        log.info("Video capture stopped.");
    }
}