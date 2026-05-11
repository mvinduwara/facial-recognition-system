package com.facerec.api;

import com.facerec.capture.VideoCaptureService;
import com.facerec.model.RecognitionResult;
import com.facerec.recognition.FaceMatcher;
import com.facerec.recognition.FeatureExtractor;
import lombok.RequiredArgsConstructor;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

@RestController
@RequestMapping("/api/face")
@RequiredArgsConstructor
public class FaceController {

    private final VideoCaptureService captureService;
    private final FaceMatcher faceMatcher;
    private final FeatureExtractor featureExtractor;

    @PostMapping("/capture/start")
    public ResponseEntity<Map<String, String>> startCapture(
            @RequestParam(defaultValue = "0") String source) {
        captureService.startCapture(source);
        return ResponseEntity.ok(Map.of("status", "started", "source", source));
    }

    @PostMapping("/capture/stop")
    public ResponseEntity<Map<String, String>> stopCapture() {
        captureService.stopCapture();
        return ResponseEntity.ok(Map.of("status", "stopped"));
    }

    @PostMapping("/register")
    public ResponseEntity<Map<String, String>> registerFace(
            @RequestParam("image") MultipartFile image,
            @RequestParam("personId") String personId,
            @RequestParam("name") String name) throws Exception {

        Path tempFile = Files.createTempFile("face_", ".jpg");
        image.transferTo(tempFile.toFile());

        Mat mat = Imgcodecs.imread(tempFile.toString());

        float[] embedding = featureExtractor.extract(mat);

        faceMatcher.register(personId, name, embedding);
        Files.deleteIfExists(tempFile);

        return ResponseEntity.ok(Map.of(
                "status", "registered",
                "personId", personId,
                "name", name
        ));
    }

    @PostMapping("/recognize")
    public ResponseEntity<RecognitionResult> recognizeFace(
            @RequestParam("image") MultipartFile image) throws Exception {

        Path tempFile = Files.createTempFile("face_", ".jpg");
        image.transferTo(tempFile.toFile());

        Mat mat = Imgcodecs.imread(tempFile.toString());
        float[] embedding = featureExtractor.extract(mat);
        RecognitionResult result = faceMatcher.match(embedding);

        Files.deleteIfExists(tempFile);
        return ResponseEntity.ok(result);
    }
}