package com.facerec.storage;

import com.facerec.model.FaceEmbedding;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface FaceVectorRepository extends JpaRepository<FaceEmbedding, String> {

    /**
     * Uses pgvector's <=> operator for cosine distance ANN search.
     * Returns the nearest neighbor within the similarity threshold.
     */
    @Query(value = """
            SELECT * FROM face_embeddings
            WHERE 1 - (embedding <=> CAST(:queryVector AS vector)) >= :threshold
            ORDER BY embedding <=> CAST(:queryVector AS vector)
            LIMIT 1
            """, nativeQuery = true)
    Optional<FaceEmbedding> findNearestNeighbor(
            @Param("queryVector") float[] queryVector,
            @Param("threshold") float threshold
    );
}