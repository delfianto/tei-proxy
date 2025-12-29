# TEI Proxy

A lightweight Rust proxy that bridges **Open WebUI** requests with **HuggingFace Text Embeddings Inference (TEI)** for reranking and embeddings.

There's an open [GitHub issue](https://github.com/huggingface/text-embeddings-inference/issues/683) for native OpenAI-compatible rerank support in TEI. This proxy fills that gap.

## Features

- Translates OpenWebUI/OpenAI-style rerank requests to TEI format
- Passthrough proxy for embeddings (preserves upstream responses)
- Configurable batch limits, timeouts, and auth
- Request ID tracking for debugging
- Deep health checks with upstream status
- Graceful shutdown on SIGINT
- CORS-enabled for browser clients

## Configuration

| Variable                | Default                      | Description                        |
| ----------------------- | ---------------------------- | ---------------------------------- |
| `TEI_ENDPOINT`          | `http://localhost:4000`      | Legacy fallback for both services  |
| `RERANKING_HOST`        | (falls back to TEI_ENDPOINT) | Rerank service URL                 |
| `RERANKING_API_KEY`     | (none)                       | Bearer token for rerank service    |
| `EMBEDDING_HOST`        | (falls back to TEI_ENDPOINT) | Embedding service URL              |
| `EMBEDDING_API_KEY`     | (none)                       | Bearer token for embedding service |
| `TEI_PROXY_PORT`        | `8000`                       | Port this proxy listens on         |
| `MAX_CLIENT_BATCH_SIZE` | `1000`                       | Max documents per rerank request   |
| `RERANK_TIMEOUT_SECS`   | `30`                         | Timeout for rerank requests        |
| `EMBED_TIMEOUT_SECS`    | `60`                         | Timeout for embedding requests     |

**Auth priority:** Environment API key > Client `Authorization` header

## Running

```bash
# Development
RUST_LOG=debug cargo run

# Production
RERANKING_HOST="http://tei-rerank:4000" \
EMBEDDING_HOST="http://tei-embed:4000" \
MAX_CLIENT_BATCH_SIZE=500 \
cargo run --release
```

## API

### Health Check

```
GET /health
```

Returns upstream connectivity status:

```json
{
    "status": "healthy",
    "service": "tei-proxy",
    "upstreams": { "rerank": true, "embed": true },
    "supported_endpoints": [
        "/v1/rerank",
        "/v1/embeddings",
        "/rerank",
        "/embeddings"
    ]
}
```

### Rerank

```
POST /rerank
POST /v1/rerank
```

**Request (OpenWebUI format):**

```json
{
    "query": "example search",
    "documents": ["doc1", "doc2", "doc3"],
    "model": "optional-ignored",
    "top_n": 2
}
```

**Response:**

```json
{
    "results": [
        { "index": 1, "relevance_score": 0.87 },
        { "index": 0, "relevance_score": 0.42 }
    ]
}
```

### Embeddings

```
POST /embeddings
POST /v1/embeddings
```

Pure passthrough to upstream TEI `/v1/embeddings` endpoint. Request and response formats are preserved.

## Testing

```bash
cargo test
```

Tests use [wiremock](https://github.com/LukeMathWalker/wiremock-rs) for mocking upstream services.

## License

MIT
