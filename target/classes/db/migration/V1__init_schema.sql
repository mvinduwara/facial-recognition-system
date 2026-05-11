CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS face_embeddings (
                                               id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id VARCHAR(100) NOT NULL,
    name VARCHAR(255) NOT NULL,
    embedding VECTOR(128) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
    );

CREATE INDEX IF NOT EXISTS face_embedding_cosine_idx
    ON face_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE TABLE IF NOT EXISTS recognition_events (
                                                  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id VARCHAR(100),
    recognized BOOLEAN NOT NULL,
    confidence SMALLINT,
    emotion VARCHAR(50),
    timestamp TIMESTAMPTZ DEFAULT NOW()
    );

CREATE INDEX IF NOT EXISTS recognition_events_person_idx ON recognition_events (person_id);
CREATE INDEX IF NOT EXISTS recognition_events_time_idx ON recognition_events (timestamp DESC);