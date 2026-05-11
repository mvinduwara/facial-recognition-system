package com.facerec.analysis;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

@Slf4j
@Component
public class EmotionAnalyzer {

    private MultiLayerNetwork emotionModel;
    private static final String[] EMOTIONS = {
            "Angry", "Disgusted", "Fearful",
            "Happy", "Neutral", "Sad", "Surprised"
    };

    @PostConstruct
    public void loadModel() throws Exception {
        emotionModel = ModelSerializer.restoreMultiLayerNetwork(
                getClass().getClassLoader().getResourceAsStream("models/emotion_model.zip")
        );
        log.info("Emotion classification model loaded ({} classes).", EMOTIONS.length);
    }

    public String analyze(Mat faceROI) {
        Mat gray = new Mat();
        Imgproc.cvtColor(faceROI, gray, Imgproc.COLOR_BGR2GRAY);
        Mat resized = new Mat();
        Imgproc.resize(gray, resized, new Size(48, 48));

        float[] pixels = new float[48 * 48];
        byte[] raw = new byte[48 * 48];
        resized.get(0, 0, raw);

        for (int i = 0; i < raw.length; i++) {
            pixels[i] = (raw[i] & 0xFF) / 255.0f;
        }

        INDArray input = Nd4j.create(pixels, new int[]{1, 1, 48, 48});
        INDArray output = emotionModel.output(input);

        int maxIdx = 0;
        float maxVal = 0;
        for (int i = 0; i < EMOTIONS.length; i++) {
            float val = output.getFloat(i);
            if (val > maxVal) {
                maxVal = val;
                maxIdx = i;
            }
        }
        return EMOTIONS[maxIdx];
    }
}