package com.facerec.recognition;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.io.InputStream;

@Slf4j
@Component
public class FeatureExtractor {

    private ComputationGraph faceNetModel;
    private static final int FACE_SIZE = 160;
    private static final int EMBEDDING_SIZE = 128;

    @PostConstruct
    public void loadModel() throws Exception {
        try (InputStream is = getClass().getClassLoader().getResourceAsStream("models/facenet_model.zip")) {
            faceNetModel = ModelSerializer.restoreComputationGraph(is, true);
            log.info("FaceNet model loaded. Input: {}x{}, Output: {} dims", FACE_SIZE, FACE_SIZE, EMBEDDING_SIZE);
        }
    }

    public float[] extract(Mat faceROI) {
        Mat resized = new Mat();
        Imgproc.resize(faceROI, resized, new Size(FACE_SIZE, FACE_SIZE));

        float[] pixelData = matToFloatArray(resized);
        INDArray input = Nd4j.create(pixelData, new int[]{1, 3, FACE_SIZE, FACE_SIZE});

        input = input.div(127.5f).sub(1.0f);

        INDArray[] outputs = faceNetModel.output(input);
        INDArray embedding = outputs[0];

        embedding = Transforms.normalizeVector(embedding);
        return embedding.toFloatVector();
    }

    private float[] matToFloatArray(Mat mat) {
        int height = mat.rows();
        int width = mat.cols();
        float[] data = new float[3 * height * width];
        byte[] bgr = new byte[3 * height * width];
        mat.get(0, 0, bgr);

        for (int i = 0; i < height * width; i++) {
            data[0 * height * width + i] = (bgr[i * 3 + 2] & 0xFF); // R
            data[1 * height * width + i] = (bgr[i * 3 + 1] & 0xFF); // G
            data[2 * height * width + i] = (bgr[i * 3] & 0xFF);     // B
        }
        return data;
    }
}