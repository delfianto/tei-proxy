# Embedding Dialects

Embeddings are the easy half of this proxy: unlike rerank, TEI already ships
an OpenAI-compatible route natively, so the proxy only passes bytes through.
This document compares the dialects anyway, because the *differences* explain
two things that look arbitrary from the outside: why the proxy exposes two
spellings of the same route, and why no translation shim is needed.

See [README.md](README.md) for the full TEI native reference.

## The dialects at a glance

| | TEI native `/embed` | OpenAI style (`/v1/embeddings`) | Ollama (`/api/embed`) |
| --- | --- | --- | --- |
| Input field | `inputs` | `input` | `input` |
| Model field | ✗ (one model per instance) | `model` (TEI ignores it) | `model` (selects model) |
| Response | bare 2D float array | `{object, data[].embedding, model, usage}` | `{embeddings: [[...]], ...}` |
| Usage/token counts | ✗ | ✓ | ✓ |
| Extras | `normalize`, `prompt_name`, `truncation_direction` | `encoding_format`, `dimensions` | `truncate` |
| Spoken by | TEI-specific clients | OpenAI SDKs, Open WebUI `openai` engine, LangChain, LlamaIndex | Ollama clients, Open WebUI `ollama` engine |

## Why a passthrough, not a shim

TEI's native `/embed` returns a bare `[[floats]]` array that no OpenAI-style
client can parse — structurally the same problem as rerank. But TEI *also*
implements `POST /v1/embeddings` in the full OpenAI shape (envelope, `data`
ordered by `index`, real `usage` counts). Since the standard dialect already
exists upstream, translation would only add a place for bugs. The proxy
forwards the request body, response bytes, and error status codes untouched.

What the proxy still adds on this path:

- a single origin for Open WebUI to talk to (rerank + embeddings behind one
  host/port),
- auth injection (`EMBEDDING_API_KEY` overrides the client's `Authorization`
  header),
- an independent timeout (`EMBED_TIMEOUT_SECS`, default 60s — ingestion
  batches are slower than queries),
- the dual route spelling below.

## Why two spellings of the route

Open WebUI's OpenAI embedding engine builds its target URL as
`f"{base_url}/embeddings"` — it **hard-appends** the path
(`generate_openai_batch_embeddings` in `retrieval/utils.py`). Contrast the
external reranker, which POSTs to its configured URL verbatim. Both spellings
must therefore exist:

| Open WebUI base URL | It posts to | Caught by proxy route |
| --- | --- | --- |
| `http://tei-proxy:8000/v1` | `/v1/embeddings` | `/v1/embeddings` |
| `http://tei-proxy:8000` | `/embeddings` | `/embeddings` |

A trailing slash on the base URL produces `/v1//embeddings`, which matches no
route — this is the most common misconfiguration.

## What Open WebUI actually sends

Engine `openai` (`RAG_EMBEDDING_ENGINE=openai`):

```json
{
    "input": ["chunk one", "chunk two"],
    "model": "<RAG_EMBEDDING_MODEL>"
}
```

with `Authorization: Bearer <RAG_OPENAI_API_KEY>`. It reads
`data[].embedding` in array order and raises if the `data` key is missing.
Chunks are batched `RAG_EMBEDDING_BATCH_SIZE` at a time (default **1** — one
HTTP request per chunk; raise it, and keep TEI `--max-client-batch-size`
above it).

**The prefix trap:** if `RAG_EMBEDDING_PREFIX_FIELD_NAME` is set, Open WebUI
moves the query/content prefix out of the text into a custom JSON field
(meant for APIs like Cohere's `input_type`). TEI silently ignores unknown
fields, so the prefix is **dropped without any error**. With TEI, leave that
variable unset and use `RAG_EMBEDDING_QUERY_PREFIX` /
`RAG_EMBEDDING_CONTENT_PREFIX`, which are prepended into the text itself.
(TEI-native `prompt_name` would be the cleaner mechanism, but nothing in the
OpenAI dialect carries it.)

## The Ollama dialect (not supported)

Open WebUI's `ollama` engine posts to `{url}/api/embed` with
`{"input": [...], "model": "...", "truncate": true}` and reads a top-level
`embeddings` key. Neither TEI nor this proxy implements that route — point
Open WebUI's *openai* engine at the proxy instead. (An `/api/embed` →
`/v1/embeddings` shim would be straightforward if TEI ever needs to
impersonate Ollama for other tools.)

Setup instructions live in the main [README](../README.md#openwebui-setup-1).
