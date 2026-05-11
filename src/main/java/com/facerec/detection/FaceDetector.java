package com.facerec.detection;

import lombok.extern.slf4j.Slf4j;
import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.ArrayList;
import java.util.List;

@Slf4j
@Component
public class FaceDetector {

    private Net dnnNet;
    private static final double CONFIDENCE_THRESHOLD = 0.7;
    private static final int INPUT_SIZE = 300;

    @PostConstruct
    public void init() {
        String prototxt = getClass().getClassLoader().getResource("models/deploy.prototxt").getPath();
        String caffeModel = getClass().getClassLoader().getResource("models/res10_300x300_ssd.caffemodel").getPath();

        if (System.getProperty("os.name").toLowerCase().contains("win") && prototxt.startsWith("/")) {
            prototxt = prototxt.substring(1);
            caffeModel = caffeModel.substring(1);
        }

        dnnNet = Dnn.readNetFromCaffe(prototxt, caffeModel);
        dnnNet.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
        dnnNet.setPreferableTarget(Dnn.DNN_TARGET_CPU);

        log.info("Face detection DNN model loaded successfully.");
    }

    public List<Rect> detect(Mat frame) {
        List<Rect> faces = new ArrayList<>();

        Mat blob = Dnn.blobFromImage(
                frame,
                1.0,
                new Size(INPUT_SIZE, INPUT_SIZE),
                new Scalar(104.0, 177.0, 123.0),
                false,
                false
        );

        dnnNet.setInput(blob);
        Mat detections = dnnNet.forward();

        Mat detectionMat = detections.reshape(1, (int) detections.total() / 7);
        int frameWidth = frame.cols();
        int frameHeight = frame.rows();

        for (int i = 0; i < detectionMat.rows(); i++) {
            double confidence = detectionMat.get(i, 2)[0];
            if (confidence > CONFIDENCE_THRESHOLD) {
                int x1 = (int) (detectionMat.get(i, 3)[0] * frameWidth);
                int y1 = (int) (detectionMat.get(i, 4)[0] * frameHeight);
                int x2 = (int) (detectionMat.get(i, 5)[0] * frameWidth);
                int y2 = (int) (detectionMat.get(i, 6)[0] * frameHeight);

                x1 = Math.max(0, x1);
                y1 = Math.max(0, y1);
                x2 = Math.min(frameWidth, x2);
                y2 = Math.min(frameHeight, y2);

                faces.add(new Rect(x1, y1, x2 - x1, y2 - y1));
            }
        }
        return faces;
    }
}