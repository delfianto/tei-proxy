# Rerank Dialects

There is no standard rerank API. Every vendor invented their own, TEI invented
a fourth, and Open WebUI's client has a parsing quirk that deserves its own
endpoint. This document lays out each dialect's structure against TEI's native
`/rerank` (the format the proxy actually calls) and explains why each shim
exists.

See [README.md](README.md) for the full TEI native reference.

## The problem at a glance

| | TEI native `/rerank` | Jina | Cohere v2 | OpenWebUI client |
| --- | --- | --- | --- | --- |
| Documents field | `texts` | `documents` | `documents` | `documents` |
| Score field | `score` | `relevance_score` | `relevance_score` | reads `relevance_score` |
| Response envelope | **bare array** | `{model, usage, results}` | `{id, results, meta}` | reads `{results}` |
| `top_n` | ✗ (no such field) | ✓ | ✓ | sends `len(documents)` |
| Result order | score desc | score desc | score desc | re-sorts by `index` itself |
| Return doc text | `return_text` | `return_documents` | ✗ (removed in v2) | ignores it |

No client that speaks Jina, Cohere, or the Open WebUI format can talk to TEI's
`/rerank` directly: the request field is named differently (`texts`), the
response has no `results` wrapper, and the score key is wrong. That triple
mismatch is the entire reason this proxy exists (TEI issue #683; fix pending in
unmerged PR #797, whose schemas the proxy mirrors).

## Shared translation core

All three proxy dialects share one pipeline (`handle_rerank` in
`src/handlers.rs`):

1. Validate: non-empty `query`, non-empty `documents`, batch ≤
   `MAX_CLIENT_BATCH_SIZE`.
2. Extract text per document: plain string → itself; object → `page_content`
   key, then `text` key (legacy Open WebUI / LangChain shapes); otherwise the
   raw JSON as a string.
3. Call TEI native: `{query, texts, truncate}` (truncate defaults to `true`).
4. Shape TEI's `[{index, score}]` into the dialect's response.

Only step 4 differs per dialect.

## Jina dialect — `POST /jina/rerank` (alias: `/v1/rerank`)

The de-facto open standard: Jina SDKs, LangChain's `JinaRerank`, and most
"OpenAI-style reranker" integrations speak it. The alias `/v1/rerank` is the
path Jina SDKs hardcode — and the exact path TEI PR #797 will expose, so
clients pointed at the alias migrate to native TEI transparently.

**Request** (proxy is more lenient than TEI PR #797, which requires `model`):

```json
{
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "example search",
    "documents": ["doc1", "doc2", "doc3"],
    "top_n": 2,
    "return_documents": true
}
```

**Response** — relevance-sorted, truncated to `top_n`:

```json
{
    "object": "rerank",
    "model": "BAAI/bge-reranker-v2-m3",
    "usage": { "prompt_tokens": 12, "total_tokens": 12 },
    "results": [
        { "index": 1, "relevance_score": 0.87, "document": { "text": "doc2" } },
        { "index": 0, "relevance_score": 0.42, "document": { "text": "doc1" } }
    ]
}
```

Translation rules:

- `documents` → TEI `texts`; TEI `score` → `relevance_score`.
- `top_n` applied proxy-side after sorting (TEI has no such field).
- `return_documents: true` → attach `document.text` from the *original* input
  texts (defaults to `false`, matching TEI PR #797).
- `usage` is **estimated** at ~4 chars/token, counting the query once per
  document pair — TEI's `/rerank` reports no token counts.
- `model` is echoed back, never validated — TEI serves whatever it loaded.

## Cohere dialect — `POST /cohere/rerank` (alias: `/v2/rerank`)

Cohere's v2 Rerank API. The alias `/v2/rerank` is the path Cohere SDKs
hardcode, and again matches TEI PR #797. Cohere v2 dropped `return_documents`;
results never carry document text.

**Request** — same shape as Jina; `max_tokens_per_doc` is accepted and ignored
(per-document truncation is delegated to TEI `--auto-truncate`).

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

- `id` is the proxy's per-request UUID (also used in logs — grep for it).
- `meta` is static compatibility filler for SDKs that dereference it.

## OpenWebUI dialect — `POST /openwebui/rerank`

Open WebUI's `ExternalReranker`
(`backend/open_webui/retrieval/models/external.py`) sends a Jina-ish request:

```json
{
    "model": "<RAG_RERANKING_MODEL>",
    "query": "user query",
    "documents": ["chunk1", "chunk2", "chunk3"],
    "top_n": 3
}
```

with `Authorization: Bearer <RAG_EXTERNAL_RERANKER_API_KEY>`, POSTed to the
configured URL **verbatim** (no path appended). It parses only
`results[].{index, relevance_score}`.

**The footgun that justifies a dedicated shim:** Open WebUI sorts the results
by `index`, extracts the scores into a list, and **zips that list positionally
against its own document list** (`RerankCompressor` in
`retrieval/utils.py`). It always sends `top_n = len(documents)`, but if a
reranker ever returns fewer results than documents — honoring a smaller
`top_n`, deduplicating, erroring on one document — the zip silently pairs
scores with the wrong documents. No error is raised; RAG quality just quietly
corrupts. Example: 4 documents, results for indices `[0, 1, 3]` → doc 2
receives doc 3's score and doc 3 receives nothing.

The shim therefore guarantees what Open WebUI actually needs rather than what
it asks for:

- **`top_n` is deliberately ignored** — every document gets a result.
- Results are ordered by `index` (Open WebUI re-sorts anyway; this makes the
  contract obvious).
- The response is the bare `{"results": [...]}` with no extra keys:

```json
{
    "results": [
        { "index": 0, "relevance_score": 0.42 },
        { "index": 1, "relevance_score": 0.87 },
        { "index": 2, "relevance_score": 0.13 }
    ]
}
```

If any handler change would make this endpoint return fewer results than
documents sent, it is a bug. The Jina dialect also works with Open WebUI
(extra response keys are ignored, and Open WebUI's own `top_n` covers all
documents) — the shim exists to make the full-coverage guarantee structural
instead of accidental.

Setup instructions live in the main [README](../README.md#openwebui-setup).
