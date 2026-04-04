CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS incoming_news (
    id UUID PRIMARY KEY,
    external_id TEXT UNIQUE NOT NULL,
    title TEXT NULL,
    source TEXT NULL,
    published_at TIMESTAMPTZ NULL,
    raw_text TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'new',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error_message TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_incoming_news_status_created_at
    ON incoming_news (status, created_at);

CREATE TABLE IF NOT EXISTS topic_metadata (
    cluster_id INTEGER PRIMARY KEY,
    label TEXT NOT NULL,
    top_words JSONB NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS clustering_reports (
    id UUID PRIMARY KEY,
    news_id UUID NOT NULL REFERENCES incoming_news(id) ON DELETE CASCADE,
    cluster_id INTEGER NULL,
    cluster_label TEXT NOT NULL,
    top_words_snapshot JSONB NOT NULL,
    score DOUBLE PRECISION NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_clustering_reports_news_id
    ON clustering_reports (news_id);
