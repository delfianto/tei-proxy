# TEI Proxy

A lightweight Rust proxy that bridges **Open WebUI** requests with **HuggingFace Text Embeddings Inference (TEI)** for reranking and embeddings.

There's an open [GitHub issue](https://github.com/huggingface/text-embeddings-inference/issues/683) for standard rerank API support in TEI, with an unmerged implementation in [PR #797](https://github.com/huggingface/text-embeddings-inference/pull/797) (milestone v1.10.0) that adds Jina-compatible `/v1/rerank` and Cohere-compatible `/v2/rerank` routes. This proxy fills that gap until it ships, exposing the same API surface so clients migrate transparently.

Deeper reference material lives in [`docs/`](docs/): the native TEI API structure ([docs/README.md](docs/README.md)) and per-dialect comparisons explaining why each shim exists ([rerank](docs/RERANK.md), [embeddings](docs/EMBEDDING.md)).

## Features

- Rerank endpoints named after the dialect they speak: `/openwebui/rerank`, `/jina/rerank`, `/cohere/rerank`
- Wire-compat aliases `/v1/rerank` (Jina) and `/v2/rerank` (Cohere) — the paths vendor SDKs and TEI PR #797 use
- Strict OpenWebUI shim guarantees one score per document (see below)
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

### TEI Requirements
When running the HuggingFace TEI docker container alongside this proxy, ensure you set `--max-client-batch-size` to be greater than or equal to the proxy's `MAX_CLIENT_BATCH_SIZE` (default 1000) to prevent `413 Payload Too Large` errors. You should also enable `--auto-truncate`.

```bash
docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:1.9.2 \
    --model-id BAAI/bge-reranker-base \
    --max-client-batch-size 1000 \
    --auto-truncate
```

### Proxy

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

For container healthchecks, `tei-proxy --healthcheck` probes this endpoint
in-process and exits 0/1 — the runtime image is `FROM scratch`, so the binary
itself is the only available HTTP client. The Dockerfile ships a matching
`HEALTHCHECK` instruction, so Docker reports container health out of the box.

Returns upstream connectivity status:

```json
{
    "status": "healthy",
    "service": "tei-proxy",
    "upstreams": { "rerank": true, "embed": true },
    "endpoints": {
        "rerank": {
            "openwebui": ["/openwebui/rerank"],
            "jina": ["/jina/rerank", "/v1/rerank"],
            "cohere": ["/cohere/rerank", "/v2/rerank"]
        },
        "embeddings": ["/v1/embeddings", "/embeddings"]
    }
}
```

### Endpoint naming

One rerank engine, three dialects. The canonical path tells you which response
shape you get; the aliases exist because third-party SDKs hardcode their
vendor's path:

| Canonical            | Alias         | Why the alias exists                          |
| -------------------- | ------------- | --------------------------------------------- |
| `/openwebui/rerank`  | —             | OpenWebUI posts to whatever URL you configure |
| `/jina/rerank`       | `/v1/rerank`  | Jina SDKs post to `/v1/rerank`                |
| `/cohere/rerank`     | `/v2/rerank`  | Cohere SDKs post to `/v2/rerank`              |

`/v1/rerank` and `/v2/rerank` are also exactly what TEI will expose natively
once PR #797 ships, so clients pointed at the aliases migrate transparently.

**How OpenWebUI builds its request URLs** (this asymmetry is why the embedding
routes come in two spellings but the rerank shim only needs one):

| OpenWebUI engine        | URL behavior                                            |
| ----------------------- | ------------------------------------------------------- |
| External reranker       | POSTs to the configured URL **verbatim**                |
| OpenAI embedding engine | **hard-appends `/embeddings`** to the configured base URL |

### Rerank (Jina dialect)

```
POST /jina/rerank      (alias: /v1/rerank)
```

Mirrors the [Jina Reranker API](https://jina.ai/reranker/) as implemented by TEI PR #797.

**Request:**

```json
{
    "model": "optional-echoed-back",
    "query": "example search",
    "documents": ["doc1", "doc2", "doc3"],
    "top_n": 2,
    "return_documents": true
}
```

`documents` entries may be plain strings or objects carrying `page_content` or
`text` keys (legacy OpenWebUI/LangChain shapes). `return_documents` defaults to
`false`, matching TEI PR #797.

**Response** (results sorted by relevance, limited to `top_n`):

```json
{
    "object": "rerank",
    "model": "optional-echoed-back",
    "usage": { "prompt_tokens": 12, "total_tokens": 12 },
    "results": [
        { "index": 1, "relevance_score": 0.87, "document": { "text": "doc2" } },
        { "index": 0, "relevance_score": 0.42, "document": { "text": "doc1" } }
    ]
}
```

Token counts are estimated (~4 chars/token); TEI's `/rerank` does not report usage.

### Rerank (Cohere dialect)

```
POST /cohere/rerank    (alias: /v2/rerank)
```

Same request shape (Cohere's `max_tokens_per_doc` is accepted and ignored —
truncation is delegated to TEI's `--auto-truncate`).

**Response:**

```json
{
    "id": "3d1b03f2-6a4b-4f8f-9c8e-0f3a4c5d6e7f",
    "results": [
        { "index": 1, "relevance_score": 0.87 },
        { "index": 0, "relevance_score": 0.42 }
    ],
    "meta": {
        "api_version": { "version": "2" },
        "billed_units": { "search_units": 1 }
    }
}
```

### Rerank (OpenWebUI dialect)

```
POST /openwebui/rerank
```

Strict shim for OpenWebUI's `ExternalReranker`. OpenWebUI sends
`{model, query, documents, top_n}` and **zips the returned scores positionally
against its documents** — if fewer results come back than documents were sent,
context silently misaligns. This endpoint therefore:

- always returns one result per document (`top_n` is deliberately ignored;
  OpenWebUI sends `top_n = len(documents)` anyway and re-sorts by score itself)
- returns results ordered by `index`
- responds with the bare shape OpenWebUI parses, nothing else:

```json
{
    "results": [
        { "index": 0, "relevance_score": 0.42 },
        { "index": 1, "relevance_score": 0.87 },
        { "index": 2, "relevance_score": 0.13 }
    ]
}
```

#### OpenWebUI setup

Admin Panel → Settings → Documents → **Reranking Engine: External**, or via env:

```bash
RAG_RERANKING_ENGINE=external
RAG_EXTERNAL_RERANKER_URL=http://tei-proxy:8000/openwebui/rerank
RAG_EXTERNAL_RERANKER_API_KEY=whatever-your-proxy-forwards  # sent as Bearer token
RAG_RERANKING_MODEL=any-label  # informational; TEI serves whatever model it loaded
```

OpenWebUI posts to the configured URL verbatim (no path is appended). The Jina
dialect endpoints also work with OpenWebUI — the shim just adds the
full-coverage guarantee and a minimal payload.

### Embeddings (OpenAI dialect)

```
POST /v1/embeddings
POST /embeddings
```

Pure passthrough to upstream TEI `/v1/embeddings`. TEI speaks the OpenAI
embeddings format natively, so no translation happens — request body, response
bytes, and error status codes are all preserved.

**Why two spellings?** OpenWebUI's `openai` embedding engine builds its target
as `f"{base_url}/embeddings"` — it hard-appends `/embeddings` to whatever base
URL you configure. Both routes exist so either base URL works:

| OpenWebUI base URL         | OpenWebUI posts to | Caught by         |
| -------------------------- | ------------------ | ----------------- |
| `http://tei-proxy:8000/v1` | `/v1/embeddings`   | `/v1/embeddings`  |
| `http://tei-proxy:8000`    | `/embeddings`      | `/embeddings`     |

**Request** (what OpenWebUI sends, and what TEI accepts):

```json
{
    "input": ["chunk one", "chunk two"],
    "model": "BAAI/bge-m3"
}
```

TEI also accepts `encoding_format` and `dimensions`; `model` and `user` are
accepted and ignored (TEI serves whatever model it loaded). Unknown fields are
silently ignored by TEI — see the prefix gotcha below.

**Response** (TEI native, passed through untouched):

```json
{
    "object": "list",
    "data": [
        { "object": "embedding", "embedding": [0.1, 0.2], "index": 0 },
        { "object": "embedding", "embedding": [0.3, 0.4], "index": 1 }
    ],
    "model": "BAAI/bge-m3",
    "usage": { "prompt_tokens": 12, "total_tokens": 12 }
}
```

OpenWebUI parses `data[].embedding` in array order, which TEI guarantees.

#### OpenWebUI setup

Admin Panel → Settings → Documents → **Embedding Model Engine: OpenAI**, or via env:

```bash
RAG_EMBEDDING_ENGINE=openai
RAG_OPENAI_API_BASE_URL=http://tei-proxy:8000/v1   # NO trailing slash
RAG_OPENAI_API_KEY=whatever-your-proxy-forwards     # sent as Bearer token
RAG_EMBEDDING_MODEL=BAAI/bge-m3                     # informational for TEI
RAG_EMBEDDING_BATCH_SIZE=32                         # default is 1 — raise it!
```

Gotchas (all verified against OpenWebUI source):

- **`RAG_EMBEDDING_BATCH_SIZE` defaults to 1**, meaning one HTTP request per
  chunk during document ingestion — painfully slow. Raise it (32–64), and keep
  TEI's `--max-client-batch-size` at or above it. The proxy's
  `MAX_CLIENT_BATCH_SIZE` only gates rerank; embedding batch limits are
  enforced by TEI itself.
- **No trailing slash** on the base URL: `http://tei-proxy:8000/v1/` produces
  `/v1//embeddings`, which matches no route.
- **Leave `RAG_EMBEDDING_PREFIX_FIELD_NAME` unset.** When set, OpenWebUI moves
  the query/content prefix into a custom JSON field instead of the text; TEI
  silently ignores unknown fields, so the prefix would be dropped. For models
  needing e5/bge-style prefixes, use `RAG_EMBEDDING_QUERY_PREFIX` and
  `RAG_EMBEDDING_CONTENT_PREFIX` — those are prepended into the text itself
  and survive the trip.
- OpenWebUI's `ollama` embedding engine posts to `{url}/api/embed` with a
  different wire format — use the `openai` engine with this proxy.

## Testing

Three layers, from pure functions up to real models:

```bash
# 1+2. Unit tests + full-router tests against wiremock upstreams.
#      Single-threaded because the config tests mutate global env vars.
cargo test -- --test-threads=1

# 3. Live integration tests against real TEI instances
#    (embedding at localhost:4001, reranker at localhost:4002 —
#     override with LIVE_TEI_EMBED / LIVE_TEI_RERANK)
cargo test --test live_tei -- --ignored
```

- **Unit tests** (`src/types.rs`, `src/handlers.rs`) cover config parsing,
  serde shapes for every dialect, document-text extraction, and the pure
  result-shaping helpers.
- **Router tests** (`src/routes.rs`) drive the real warp router in-process
  via `warp::test` against [wiremock](https://github.com/LukeMathWalker/wiremock-rs)
  upstreams, asserting exact response bodies per dialect, alias equivalence,
  validation errors, auth precedence, passthrough fidelity, and health states.
- **Live tests** (`tests/live_tei.rs`, `#[ignore]` by default) first probe
  both TEI `/health` endpoints and fail with a clear message if either is
  unreachable, verify via `/info` that :4001 actually serves an embedding
  model and :4002 a reranker (catches swapped ports), then exercise every
  dialect against the real models — including semantic sanity checks (the
  relevant document must outscore filler; paraphrase embeddings must be
  closer than unrelated text) and a full end-to-end run of the compiled
  proxy binary over real HTTP.

## License

MIT
