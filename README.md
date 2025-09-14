# Rerank Proxy

A lightweight Rust proxy that bridges **Open WebUI** requests with **Huggingface Text Embeddings Inference (TEI)** rerank service.
It normalizes request/response formats and enforces batch limits for the API.

There's an open [GitHub issue](https://github.com/huggingface/text-embeddings-inference/issues/683) for this, and while waiting for the final implementation from the TEI maintainer; I decide to wrote this small proxy.

---

## ✨ Features

- Accepts OpenWebUI-style rerank requests (`query` + `documents`).
- Transforms requests into TEI-compatible format.
- Validates input (non-empty query, non-empty documents).
- Enforces configurable max batch size (`MAX_CLIENT_BATCH_SIZE`).
- Handles TEI errors gracefully (timeouts, bad responses, mismatches).
- Provides structured JSON error responses.
- Includes `/health` endpoint for readiness checks.
- CORS-enabled for browser-based clients.
- Logging via `env_logger`.

---

## ⚙️ Configuration

Set the following environment variables:

| Variable                | Default                 | Description                                     |
| ----------------------- | ----------------------- | ----------------------------------------------- |
| `TEI_ENDPOINT`          | `http://localhost:4000` | Base URL of the TEI service                     |
| `TEI_PROXY_PORT`        | `8000`                  | Port where this proxy will listen               |
| `MAX_CLIENT_BATCH_SIZE` | `1000`                  | Maximum allowed number of documents per request |

---

## 🚀 Running

### With Cargo

```bash
# Build & run
cargo run
```

### With Environment Variables

```bash
TEI_ENDPOINT="http://tei:4000" \
TEI_PROXY_PORT=8080 \
MAX_CLIENT_BATCH_SIZE=500 \
cargo run --release
```

---

## 📡 API

### Health Check

```
GET /health
```

Response:

```json
{
    "status": "healthy",
    "service": "rerank-proxy"
}
```

---

### Rerank

```
POST /rerank
Content-Type: application/json
```

#### Request (OpenWebUI format)

```json
{
    "query": "example search",
    "documents": ["doc1", "doc2", "doc3"],
    "model": "optional-model-name",
    "top_n": 2
}
```

#### Transformed TEI Request

```json
{
    "query": "example search",
    "texts": ["doc1", "doc2", "doc3"]
}
```

#### Response

```json
{
    "results": [
        { "index": 1, "relevance_score": 0.87 },
        { "index": 0, "relevance_score": 0.42 },
        { "index": 2, "relevance_score": 0.15 }
    ]
}
```

#### Error Example

```json
{
    "error": "bad_request",
    "message": "Documents list cannot be empty"
}
```

---

## 🛠 Development

### Prerequisites

- Rust (edition 2021+)
- Cargo
- A running TEI service (e.g. HuggingFace’s `text-embeddings-inference` Docker image)

### Logs

Enable debug logs:

```bash
RUST_LOG=debug cargo run
```

---

## 📜 License

MIT

---
