package com.facerec.model;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.Type;

import java.time.Instant;

@Entity
@Table(name = "face_embeddings",
        indexes = @Index(name = "idx_person_id", columnList = "person_id"))
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FaceEmbedding {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private String id;

    @Column(name = "person_id", nullable = false)
    private String personId;

    @Column(name = "name", nullable = false)
    private String name;

    @Column(name = "embedding", columnDefinition = "vector(128)")
    private float[] embedding;

    @Column(name = "created_at")
    private Instant createdAt = Instant.now();
}