# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. `CLAUDE.md` is a symlink to `AGENTS.md`; the same guidance applies to all AI coding agents.

## Project

Rust proxy bridging Open WebUI to HuggingFace Text Embeddings Inference (TEI): translates rerank requests into TEI's native format and passes embeddings through. It exists because TEI has no standard rerank API — tracked in TEI issue #683, implemented in unmerged PR #797 (milestone v1.10.0). The proxy deliberately mirrors PR #797's endpoint paths and response shapes so clients migrate transparently once TEI ships them natively. Re-check that PR's merge status before extending rerank features.

## Commands

```bash
cargo test -- --test-threads=1   # ALWAYS single-threaded: config tests mutate global env vars and flake in parallel
cargo test test_name -- --test-threads=1   # single test by name substring
cargo test --test live_tei -- --ignored    # live integration tests; needs real TEI (embed :4001, rerank :4002; override LIVE_TEI_EMBED / LIVE_TEI_RERANK)
cargo clippy --all-targets       # lint; keep at zero warnings
RUST_LOG=debug cargo run         # dev server (env config table in README)
./tests/test_openwebui_rerank.sh # manual e2e against a running proxy; PROXY_BASE=http://host:port overrides target
```

## Architecture

Lib + bin split so integration tests can drive the real router in-process: `src/lib.rs` exposes `routes` / `handlers` / `types`; `src/main.rs` is a thin binary (env config, logging, graceful shutdown). The flow is `routes.rs` (`build_routes`, the full warp route tree) → `handlers.rs` (logic) → `types.rs` (config + serde types). `build_routes` takes `ProxyConfig` **by value** — borrowing it makes the returned filter non-`'static` under edition 2024 RPIT capture rules.

- **One rerank engine, three dialects.** `handle_rerank` in `handlers.rs` does shared validation, document-text extraction, and the upstream call to TEI's native `/rerank`, then shapes the response per `RerankFlavor` (Jina / Cohere / OpenWebUI). Routes in `main.rs` are built by the `rerank_route(flavor)` closure: canonical paths `/openwebui/rerank`, `/jina/rerank`, `/cohere/rerank` plus wire-compat aliases `/v1/rerank` (Jina SDK path) and `/v2/rerank` (Cohere SDK path). Bare `/rerank` was removed intentionally.
- **Embeddings are a byte-for-byte passthrough** to upstream TEI `/v1/embeddings` (TEI natively speaks the OpenAI shape). Both `/v1/embeddings` and `/embeddings` must exist because Open WebUI's OpenAI embedding engine hard-appends `/embeddings` to the configured base URL.
- Rerank `documents` entries may be strings or objects with `page_content`/`text` keys (legacy Open WebUI shapes) — keep accepting both.
- Auth: an env API key (`RERANKING_API_KEY`/`EMBEDDING_API_KEY`) overrides the client's `Authorization` header; otherwise the client header is forwarded.
- Tests come in three layers: pure helpers and serde shapes in `types.rs`/`handlers.rs` `#[cfg(test)]` modules; full-router tests in `routes.rs` (`warp::test` + wiremock upstreams, asserting exact response bodies); live tests in `tests/live_tei.rs` (`#[ignore]`, need real TEI — they probe reachability first and verify model types via `/info`).

## Invariants — do not "fix" these

- `/openwebui/rerank` **ignores `top_n`** and always returns one result per document, ordered by index. Open WebUI zips returned scores positionally against its documents; returning fewer results silently misaligns RAG context. This is the whole point of the shim.
- `/v1/rerank` and `/v2/rerank` request/response schemas must stay byte-compatible with TEI PR #797 (Jina and Cohere dialects respectively).
- Rerank `usage` token counts are estimates (~4 chars/token); TEI's `/rerank` reports no counts.
- Claims in README and `docs/` about Open WebUI and TEI behavior were verified against their sources — when editing, verify against upstream rather than guessing. `docs/README.md` is the native TEI API reference; `docs/RERANK.md` and `docs/EMBEDDING.md` explain each dialect vs TEI native and the rationale for every shim decision.
