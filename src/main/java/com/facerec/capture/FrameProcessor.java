package com.facerec.capture;

import com.facerec.detection.FaceDetector;
import com.facerec.model.RecognitionResult;
import com.facerec.recognition.FeatureExtractor;
import com.facerec.recognition.FaceMatcher;
import com.facerec.analysis.EmotionAnalyzer;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.awt.image.BufferedImage;
import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class FrameProcessor {

    private final FaceDetector faceDetector;
    private final FeatureExtractor featureExtractor;
    private final FaceMatcher faceMatcher;
    private final EmotionAnalyzer emotionAnalyzer;
    private final KafkaTemplate<String, RecognitionResult> kafkaTemplate;

    private static final String TOPIC = "face-recognition-events";

    @Async("frameProcessorExecutor")
    public void processFrame(BufferedImage image) {
        try {
            Mat mat = bufferedImageToMat(image);

            List<Rect> faces = faceDetector.detect(mat);

            for (Rect face : faces) {
                Mat faceROI = new Mat(mat, face);

                float[] embedding = featureExtractor.extract(faceROI);

                RecognitionResult result = faceMatcher.match(embedding);

                String emotion = emotionAnalyzer.analyze(faceROI);
                result.setEmotion(emotion);
                result.setFaceRect(face);

                kafkaTemplate.send(TOPIC, result);

                log.debug("Processed face: {} | Confidence: {}%", result.getName(), result.getConfidence());
            }
        } catch (Exception e) {
            log.error("Frame processing error: {}", e.getMessage());
        }
    }

    private Mat bufferedImageToMat(BufferedImage bi) {
        Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
        byte[] data = new byte[bi.getWidth() * bi.getHeight() * 3];
        int[] rgb = bi.getRGB(0, 0, bi.getWidth(), bi.getHeight(), null, 0, bi.getWidth());

        for (int i = 0; i < rgb.length; i++) {
            data[i * 3] = (byte) ((rgb[i] >> 16) & 0xFF);
            data[i * 3 + 1] = (byte) ((rgb[i] >> 8) & 0xFF);
            data[i * 3 + 2] = (byte) (rgb[i] & 0xFF);
        }
        mat.put(0, 0, data);
        return mat;
    }
}