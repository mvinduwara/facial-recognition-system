package com.facerec.model;

import lombok.Data;
import org.opencv.core.Rect;

@Data
public class RecognitionResult {
    private String personId;
    private String name;
    private int confidence;
    private boolean recognized;
    private String emotion;
    private String estimatedAge;
    private String gender;
    private Rect faceRect;
    private long timestamp = System.currentTimeMillis();
}