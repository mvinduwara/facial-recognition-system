package com.facerec.recognition;

import com.facerec.model.FaceEmbedding;
import com.facerec.model.RecognitionResult;
import com.facerec.storage.FaceVectorRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Slf4j
@Service
@RequiredArgsConstructor
public class FaceMatcher {

    private final FaceVectorRepository faceVectorRepository;
    private final RedisTemplate<String, Object> redisTemplate;

    private static final float RECOGNITION_THRESHOLD = 0.75f;
    private static final String CACHE_PREFIX = "face:embed:";

    public RecognitionResult match(float[] embedding) {
        Optional<FaceEmbedding> best = faceVectorRepository
                .findNearestNeighbor(embedding, RECOGNITION_THRESHOLD);

        RecognitionResult result = new RecognitionResult();

        if (best.isPresent()) {
            FaceEmbedding match = best.get();
            float similarity = cosineSimilarity(embedding, match.getEmbedding());

            result.setPersonId(match.getPersonId());
            result.setName(match.getName());
            result.setConfidence((int) (similarity * 100));
            result.setRecognized(true);
        } else {
            result.setName("Unknown");
            result.setRecognized(false);
            result.setConfidence(0);
        }
        return result;
    }

    public void register(String personId, String name, float[] embedding) {
        FaceEmbedding entity = FaceEmbedding.builder()
                .personId(personId)
                .name(name)
                .embedding(embedding)
                .build();

        faceVectorRepository.save(entity);

        redisTemplate.delete(CACHE_PREFIX + personId);
        log.info("Registered new face: {} ({})", name, personId);
    }

    private float cosineSimilarity(float[] a, float[] b) {
        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (float) (Math.sqrt(normA) * Math.sqrt(normB));
    }
}