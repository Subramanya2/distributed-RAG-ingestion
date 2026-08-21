-- Bootstrap script executed automatically by the pgvector/pgvector Docker image
-- on first container initialization.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
