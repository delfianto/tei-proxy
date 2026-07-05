# TEI Proxy — API Documentation

This directory documents the API dialects the proxy speaks and, below, the
native HuggingFace Text Embeddings Inference (TEI) API it translates to. Every
schema here was verified against the TEI OpenAPI spec / source and the Open
WebUI source, not guessed from memory.

| Document | Contents |
| --- | --- |
| [RERANK.md](RERANK.md) | Each rerank dialect (Jina, Cohere, OpenWebUI) side by side with TEI's native `/rerank`, and why each shim exists |
| [EMBEDDING.md](EMBEDDING.md) | The OpenAI embedding dialect vs TEI's native `/embed` family, and why embeddings need a passthrough rather than a shim |

The rest of this file is a reference for **TEI's own API** — the thing on the
other side of the proxy.

## TEI serving model

A TEI instance serves exactly **one model**, set via `--model-id`. The model's
architecture determines which routes work:

- **Embedding models** (e.g. `BAAI/bge-m3`) serve `/embed`, `/embed_all`,
  `/embed_sparse`, `/v1/embeddings`.
- **Rerankers / sequence-classification models** (e.g.
  `BAAI/bge-reranker-v2-m3`) serve `/rerank` and `/predict`.

Calling a route the loaded model doesn't support returns an error. This is why
the proxy has two separately configurable upstreams (`RERANKING_HOST`,
`EMBEDDING_HOST`) — a typical deployment runs two TEI containers.

Utility routes exist on every instance: `/health`, `/info`, `/metrics`,
`/tokenize`, `/decode`.

## Native rerank: `POST /rerank`

**Request:**

```json
{
    "query": "What is Deep Learning?",
    "texts": ["Deep Learning is ...", "cheese is made of milk"],
    "raw_scores": false,
    "return_text": false,
    "truncate": false,
    "truncation_direction": "right"
}
```

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `query` | string | required | Scored against every entry in `texts` |
| `texts` | string[] | required | Note: `texts`, not `documents` — first incompatibility with every public rerank API |
| `raw_scores` | bool | `false` | `false` → sigmoid-normalized scores in [0, 1]; `true` → raw logits |
| `return_text` | bool | `false` | Echo each text back in its `Rank` |
| `truncate` | bool | `false` | The proxy sends `true` unless the client says otherwise; pair with TEI `--auto-truncate` |
| `truncation_direction` | string | `"right"` | |

**Response** — a bare JSON array (no wrapper object), sorted by score
**descending** (verified in TEI source: sort + reverse):

```json
[
    { "index": 0, "score": 0.994, "text": null },
    { "index": 1, "score": 0.021, "text": null }
]
```

The three incompatibilities that make the shims necessary:

1. Bare array — every public rerank API wraps results in `{"results": [...]}`.
2. Score field is `score` — every public API calls it `relevance_score`.
3. Request wants `texts` — every public API sends `documents`, and there is no
   `top_n`.

TEI issue [#683](https://github.com/huggingface/text-embeddings-inference/issues/683)
tracks fixing this; PR [#797](https://github.com/huggingface/text-embeddings-inference/pull/797)
(unmerged, milestone v1.10.0) adds Jina-compatible `/v1/rerank` and
Cohere-compatible `/v2/rerank` routes. This proxy's endpoints mirror that PR's
schemas exactly.

## Native embeddings: `POST /embed` family

**`/embed` request:**

```json
{
    "inputs": ["What is Deep Learning?"],
    "normalize": true,
    "truncate": false,
    "truncation_direction": "right",
    "prompt_name": null,
    "dimensions": null
}
```

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `inputs` | string \| string[] | required | |
| `normalize` | bool | `true` | L2-normalize output vectors |
| `truncate` | bool | `false` | |
| `truncation_direction` | string | `"right"` | |
| `prompt_name` | string | `null` | Named prompt from the model's Sentence Transformers config (e.g. e5 query prefix) |
| `dimensions` | int | `null` | Matryoshka truncation (newer TEI versions) |

**Response** — a bare 2D float array, no envelope, no usage:

```json
[[0.0123, -0.0456, 0.0789]]
```

Variants: `/embed_all` returns per-token vectors (3D array, no pooling);
`/embed_sparse` returns SPLADE-style sparse vectors as
`[[{"index": 12, "value": 0.5}]]`.

## OpenAI-compatible embeddings: `POST /v1/embeddings`

TEI ships this natively — it is why the proxy's embedding side is a
passthrough, not a shim.

**Request:**

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `input` | string \| string[] | required | Also accepts pre-tokenized token arrays |
| `model` | string | `null` | Accepted and **ignored** — TEI serves whatever it loaded |
| `user` | string | `null` | Accepted and ignored |
| `encoding_format` | `"float"` \| `"base64"` | `"float"` | |
| `dimensions` | int | `null` | Matryoshka truncation (newer TEI versions) |

Unknown request fields are **silently ignored** (no `deny_unknown_fields` in
TEI's deserializer) — relevant to Open WebUI's prefix-field option, see
[EMBEDDING.md](EMBEDDING.md).

**Response:**

```json
{
    "object": "list",
    "data": [
        { "object": "embedding", "embedding": [0.0123, -0.0456], "index": 0 }
    ],
    "model": "BAAI/bge-m3",
    "usage": { "prompt_tokens": 7, "total_tokens": 7 }
}
```

`data` is ordered by `index`, matching input order.

## Operational notes

- `--max-client-batch-size` (default 32) caps how many inputs one request may
  carry. Keep it ≥ the proxy's `MAX_CLIENT_BATCH_SIZE` for rerank and ≥ Open
  WebUI's `RAG_EMBEDDING_BATCH_SIZE` for embeddings, or TEI returns
  `413 Payload Too Large`.
- `--auto-truncate` makes the `truncate` flags default-on server-side; without
  it, over-length inputs error instead of truncating.
- Field availability can differ by TEI version (e.g. `dimensions` is newer);
  this reference tracks TEI `main` as of July 2026, latest release v1.9.3.
