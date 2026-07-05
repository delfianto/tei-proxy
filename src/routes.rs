use reqwest::Client;
use uuid::Uuid;
use warp::Filter;

use crate::handlers::{handle_embedding, handle_health, handle_rerank};
use crate::types::{self, ProxyConfig, RerankFlavor};

/// Build the full warp route tree. Separate from `main` so tests can drive
/// the real router in-process via `warp::test`.
pub fn build_routes(
    config: ProxyConfig,
    http_client: Client,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    let rerank_config = config.rerank.clone();
    let embed_config = config.embed.clone();
    let max_batch = config.max_batch_size;
    let rerank_timeout = config.rerank_timeout;
    let embed_timeout = config.embed_timeout;

    let rerank_ctx = warp::any().map(move || rerank_config.clone());
    let embed_ctx = warp::any().map(move || embed_config.clone());
    let client_ctx = warp::any().map(move || http_client.clone());
    let max_batch_ctx = warp::any().map(move || max_batch);
    let rerank_timeout_ctx = warp::any().map(move || rerank_timeout);
    let embed_timeout_ctx = warp::any().map(move || embed_timeout);

    let request_id = warp::any().map(|| Uuid::new_v4().to_string());
    let auth_header = warp::header::optional::<String>("authorization");

    // --- Health Check ---
    let health_rerank = config.rerank.clone();
    let health_embed = config.embed.clone();
    let health_client = Client::new();

    let health = warp::path("health").and(warp::get()).and_then(move || {
        let r = health_rerank.clone();
        let e = health_embed.clone();
        let c = health_client.clone();
        async move { handle_health(r, e, c).await }
    });

    // --- Rerank Routes ---
    let rerank_route = |flavor: RerankFlavor| {
        warp::post()
            .and(warp::body::json())
            .and(warp::any().map(move || flavor))
            .and(rerank_ctx.clone())
            .and(client_ctx.clone())
            .and(auth_header)
            .and(max_batch_ctx)
            .and(rerank_timeout_ctx)
            .and(request_id)
            .and_then(
                |req, flavor, config, client, auth, max_batch, timeout, req_id| async move {
                    handle_rerank(req, flavor, config, client, auth, max_batch, timeout, req_id)
                        .await
                },
            )
    };

    // Canonical routes: named after the API dialect they speak
    let jina_rerank = warp::path!("jina" / "rerank").and(rerank_route(RerankFlavor::Jina));
    let cohere_rerank = warp::path!("cohere" / "rerank").and(rerank_route(RerankFlavor::Cohere));
    let owui_rerank =
        warp::path!("openwebui" / "rerank").and(rerank_route(RerankFlavor::OpenWebUI));

    // Wire-compat aliases: Jina SDKs post to /v1/rerank, Cohere SDKs to
    // /v2/rerank (the same paths TEI PR #797 will expose).
    let v1_rerank = warp::path!("v1" / "rerank").and(rerank_route(RerankFlavor::Jina));
    let v2_rerank = warp::path!("v2" / "rerank").and(rerank_route(RerankFlavor::Cohere));

    // --- Embedding Routes ---
    let embed_handler = warp::post()
        .and(warp::body::json())
        .and(embed_ctx.clone())
        .and(client_ctx.clone())
        .and(auth_header)
        .and(embed_timeout_ctx)
        .and(request_id)
        .and_then(|req, config, client, auth, timeout, req_id| async move {
            handle_embedding(req, config, client, auth, timeout, req_id).await
        });

    let v1_embed = warp::path!("v1" / "embeddings").and(embed_handler.clone());
    let short_embed = warp::path!("embeddings").and(embed_handler);

    // --- CORS ---
    let cors = warp::cors()
        .allow_any_origin()
        .allow_headers(vec!["content-type", "authorization"])
        .allow_methods(vec!["GET", "POST", "OPTIONS"]);

    health
        .or(owui_rerank)
        .or(jina_rerank)
        .or(cohere_rerank)
        .or(v1_rerank)
        .or(v2_rerank)
        .or(v1_embed)
        .or(short_embed)
        .recover(types::handle_rejection)
        .with(cors)
        .with(warp::log("tei_proxy"))
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ServiceConfig;
    use std::time::Duration;
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_config(rerank_uri: &str, embed_uri: &str) -> ProxyConfig {
        ProxyConfig {
            rerank: ServiceConfig {
                endpoint: rerank_uri.to_string(),
                api_key: None,
            },
            embed: ServiceConfig {
                endpoint: embed_uri.to_string(),
                api_key: None,
            },
            port: 0,
            max_batch_size: 1000,
            rerank_timeout: Duration::from_secs(5),
            embed_timeout: Duration::from_secs(5),
        }
    }

    /// Mock TEI /rerank: three docs scored 0.5 / 0.9 / 0.3 (index order)
    async fn mock_tei_rerank() -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                {"index": 0, "score": 0.5},
                {"index": 1, "score": 0.9},
                {"index": 2, "score": 0.3}
            ])))
            .mount(&server)
            .await;
        server
    }

    async fn post_json(
        routes: &(impl Filter<Extract = (impl warp::Reply + 'static,), Error = warp::Rejection>
              + Clone
              + 'static),
        route_path: &str,
        body: &serde_json::Value,
    ) -> (u16, serde_json::Value) {
        let resp = warp::test::request()
            .method("POST")
            .path(route_path)
            .header("content-type", "application/json")
            .json(body)
            .reply(routes)
            .await;
        let status = resp.status().as_u16();
        let json = serde_json::from_slice(resp.body()).unwrap_or(serde_json::Value::Null);
        (status, json)
    }

    fn three_docs_body(top_n: Option<usize>) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": "test-model",
            "query": "q",
            "documents": ["a", "b", "c"]
        });
        if let Some(n) = top_n {
            body["top_n"] = serde_json::json!(n);
        }
        body
    }

    mod openwebui_dialect {
        use super::*;

        #[tokio::test]
        async fn full_coverage_even_when_top_n_is_smaller() {
            let tei = mock_tei_rerank().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let (status, json) =
                post_json(&routes, "/openwebui/rerank", &three_docs_body(Some(1))).await;

            assert_eq!(status, 200);
            // Exact body: every document scored, ordered by index, no extras
            assert_eq!(
                json,
                serde_json::json!({
                    "results": [
                        {"index": 0, "relevance_score": 0.5},
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 2, "relevance_score": 0.3}
                    ]
                })
            );
        }

        #[tokio::test]
        async fn exact_openwebui_wire_payload_is_accepted() {
            let tei = mock_tei_rerank().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            // Byte-for-byte what ExternalReranker.predict() sends
            let body = serde_json::json!({
                "model": "reranker",
                "query": "q",
                "documents": ["a", "b", "c"],
                "top_n": 3
            });
            let (status, json) = post_json(&routes, "/openwebui/rerank", &body).await;

            assert_eq!(status, 200);
            let results = json["results"].as_array().unwrap();
            assert_eq!(results.len(), 3);
            // OpenWebUI reads exactly these two keys per result
            for r in results {
                assert!(r["index"].is_u64());
                assert!(r["relevance_score"].is_number());
            }
        }

        #[tokio::test]
        async fn legacy_dict_documents_are_extracted() {
            let tei = MockServer::start().await;
            // Assert the proxy extracted the right text from each shape
            Mock::given(method("POST"))
                .and(path("/rerank"))
                .and(body_json(serde_json::json!({
                    "query": "q",
                    "texts": ["from page_content", "from text", "plain"],
                    "truncate": true
                })))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                    {"index": 0, "score": 0.1},
                    {"index": 1, "score": 0.2},
                    {"index": 2, "score": 0.3}
                ])))
                .mount(&tei)
                .await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let body = serde_json::json!({
                "query": "q",
                "documents": [
                    {"page_content": "from page_content", "metadata": {"name": "x.md"}},
                    {"text": "from text"},
                    "plain"
                ]
            });
            let (status, json) = post_json(&routes, "/openwebui/rerank", &body).await;

            assert_eq!(status, 200);
            assert_eq!(json["results"].as_array().unwrap().len(), 3);
        }

        #[tokio::test]
        async fn no_envelope_keys_leak() {
            let tei = mock_tei_rerank().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let (_, json) = post_json(&routes, "/openwebui/rerank", &three_docs_body(None)).await;

            let obj = json.as_object().unwrap();
            assert_eq!(obj.keys().collect::<Vec<_>>(), vec!["results"]);
        }
    }

    mod jina_dialect {
        use super::*;

        #[tokio::test]
        async fn envelope_sorting_and_top_n() {
            let tei = mock_tei_rerank().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let (status, json) = post_json(&routes, "/jina/rerank", &three_docs_body(Some(2))).await;

            assert_eq!(status, 200);
            assert_eq!(json["object"], "rerank");
            assert_eq!(json["model"], "test-model");
            assert!(json["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
            let results = json["results"].as_array().unwrap();
            assert_eq!(results.len(), 2);
            // Sorted by relevance descending
            assert_eq!(results[0]["index"], 1);
            assert_eq!(results[0]["relevance_score"], 0.9);
            assert_eq!(results[1]["index"], 0);
            assert_eq!(results[1]["relevance_score"], 0.5);
        }

        #[tokio::test]
        async fn model_defaults_when_absent() {
            let tei = mock_tei_rerank().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let body = serde_json::json!({"query": "q", "documents": ["a", "b", "c"]});
            let (_, json) = post_json(&routes, "/jina/rerank", &body).await;

            assert_eq!(json["model"], "tei-reranker");
        }

        #[tokio::test]
        async fn return_documents_attaches_original_text() {
            let tei = mock_tei_rerank().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let mut body = three_docs_body(Some(2));
            body["return_documents"] = serde_json::json!(true);
            let (_, json) = post_json(&routes, "/jina/rerank", &body).await;

            let results = json["results"].as_array().unwrap();
            // Top result is index 1 → original document "b"
            assert_eq!(results[0]["document"]["text"], "b");
            assert_eq!(results[1]["document"]["text"], "a");
        }

        #[tokio::test]
        async fn document_key_omitted_by_default() {
            let tei = mock_tei_rerank().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let (_, json) = post_json(&routes, "/jina/rerank", &three_docs_body(None)).await;

            for r in json["results"].as_array().unwrap() {
                assert!(r.get("document").is_none());
            }
        }

        #[tokio::test]
        async fn v1_alias_is_identical() {
            let tei = mock_tei_rerank().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let body = three_docs_body(Some(2));
            let (s1, j1) = post_json(&routes, "/jina/rerank", &body).await;
            let (s2, j2) = post_json(&routes, "/v1/rerank", &body).await;

            assert_eq!(s1, 200);
            assert_eq!((s1, j1), (s2, j2));
        }
    }

    mod cohere_dialect {
        use super::*;

        #[tokio::test]
        async fn envelope_shape() {
            let tei = mock_tei_rerank().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let (status, json) =
                post_json(&routes, "/cohere/rerank", &three_docs_body(Some(2))).await;

            assert_eq!(status, 200);
            assert!(!json["id"].as_str().unwrap().is_empty());
            assert_eq!(json["meta"]["api_version"]["version"], "2");
            assert_eq!(json["meta"]["billed_units"]["search_units"], 1);
            let results = json["results"].as_array().unwrap();
            assert_eq!(results.len(), 2);
            assert_eq!(results[0]["index"], 1);
            // Cohere v2 never carries document text
            assert!(results[0].get("document").is_none());
        }

        #[tokio::test]
        async fn v2_alias_matches_modulo_request_id() {
            let tei = mock_tei_rerank().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let body = three_docs_body(Some(2));
            let (s1, j1) = post_json(&routes, "/cohere/rerank", &body).await;
            let (s2, j2) = post_json(&routes, "/v2/rerank", &body).await;

            assert_eq!((s1, s2), (200, 200));
            assert_eq!(j1["results"], j2["results"]);
            assert_eq!(j1["meta"], j2["meta"]);
            assert_ne!(j1["id"], j2["id"]); // fresh UUID per request
        }

        #[tokio::test]
        async fn max_tokens_per_doc_is_tolerated() {
            let tei = mock_tei_rerank().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let mut body = three_docs_body(None);
            body["max_tokens_per_doc"] = serde_json::json!(512);
            let (status, _) = post_json(&routes, "/cohere/rerank", &body).await;

            assert_eq!(status, 200);
        }
    }

    mod routing {
        use super::*;

        #[tokio::test]
        async fn bare_rerank_route_is_gone() {
            let tei = mock_tei_rerank().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let (status, json) = post_json(&routes, "/rerank", &three_docs_body(None)).await;

            assert_eq!(status, 404);
            assert_eq!(json["error"], "not_found");
        }

        #[tokio::test]
        async fn cors_preflight_allows_browser_clients() {
            let tei = mock_tei_rerank().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let resp = warp::test::request()
                .method("OPTIONS")
                .path("/openwebui/rerank")
                .header("origin", "http://example.com")
                .header("access-control-request-method", "POST")
                .header("access-control-request-headers", "content-type")
                .reply(&routes)
                .await;

            assert_eq!(resp.status(), 200);
            assert!(resp.headers().contains_key("access-control-allow-origin"));
        }
    }

    mod validation_and_errors {
        use super::*;

        #[tokio::test]
        async fn empty_query_is_400() {
            let tei = mock_tei_rerank().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let body = serde_json::json!({"query": "  ", "documents": ["a"]});
            let (status, json) = post_json(&routes, "/openwebui/rerank", &body).await;

            assert_eq!(status, 400);
            assert_eq!(json["error"], "bad_request");
            assert_eq!(json["message"], "Query cannot be empty");
        }

        #[tokio::test]
        async fn empty_documents_is_400() {
            let tei = mock_tei_rerank().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let body = serde_json::json!({"query": "q", "documents": []});
            let (status, json) = post_json(&routes, "/jina/rerank", &body).await;

            assert_eq!(status, 400);
            assert_eq!(json["error"], "bad_request");
        }

        #[tokio::test]
        async fn batch_size_limit_is_enforced() {
            let tei = mock_tei_rerank().await;
            let mut config = test_config(&tei.uri(), &tei.uri());
            config.max_batch_size = 2;
            let routes = build_routes(config, Client::new());

            let (status, json) = post_json(&routes, "/jina/rerank", &three_docs_body(None)).await;

            assert_eq!(status, 400);
            assert!(
                json["message"]
                    .as_str()
                    .unwrap()
                    .contains("exceeds maximum 2")
            );
        }

        #[tokio::test]
        async fn malformed_json_is_400() {
            let tei = mock_tei_rerank().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let resp = warp::test::request()
                .method("POST")
                .path("/jina/rerank")
                .header("content-type", "application/json")
                .body("{not json")
                .reply(&routes)
                .await;

            assert_eq!(resp.status(), 400);
        }

        #[tokio::test]
        async fn upstream_500_maps_to_502() {
            let tei = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/rerank"))
                .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
                .mount(&tei)
                .await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let (status, json) = post_json(&routes, "/jina/rerank", &three_docs_body(None)).await;

            assert_eq!(status, 502);
            assert_eq!(json["error"], "upstream_error");
        }

        #[tokio::test]
        async fn unreachable_upstream_maps_to_502() {
            // Port 9 (discard) — nothing listens there
            let routes = build_routes(
                test_config("http://127.0.0.1:9", "http://127.0.0.1:9"),
                Client::new(),
            );

            let (status, json) = post_json(&routes, "/jina/rerank", &three_docs_body(None)).await;

            assert_eq!(status, 502);
            assert_eq!(json["error"], "upstream_error");
        }

        #[tokio::test]
        async fn malformed_upstream_body_maps_to_502() {
            let tei = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/rerank"))
                .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
                .mount(&tei)
                .await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let (status, _) = post_json(&routes, "/jina/rerank", &three_docs_body(None)).await;

            assert_eq!(status, 502);
        }
    }

    mod auth {
        use super::*;

        #[tokio::test]
        async fn client_bearer_is_forwarded() {
            let tei = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/rerank"))
                .and(header("Authorization", "Bearer owui-key"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                    {"index": 0, "score": 0.9}
                ])))
                .mount(&tei)
                .await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let resp = warp::test::request()
                .method("POST")
                .path("/openwebui/rerank")
                .header("authorization", "Bearer owui-key")
                .json(&serde_json::json!({"query": "q", "documents": ["a"]}))
                .reply(&routes)
                .await;

            assert_eq!(resp.status(), 200);
        }

        #[tokio::test]
        async fn env_key_overrides_client_header() {
            let tei = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/rerank"))
                .and(header("Authorization", "Bearer env-key"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                    {"index": 0, "score": 0.9}
                ])))
                .mount(&tei)
                .await;
            let mut config = test_config(&tei.uri(), &tei.uri());
            config.rerank.api_key = Some("env-key".to_string());
            let routes = build_routes(config, Client::new());

            let resp = warp::test::request()
                .method("POST")
                .path("/openwebui/rerank")
                .header("authorization", "Bearer client-key")
                .json(&serde_json::json!({"query": "q", "documents": ["a"]}))
                .reply(&routes)
                .await;

            assert_eq!(resp.status(), 200);
        }
    }

    mod embeddings {
        use super::*;

        fn openai_response() -> serde_json::Value {
            serde_json::json!({
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": [0.1, 0.2], "index": 0}
                ],
                "model": "test-embed",
                "usage": {"prompt_tokens": 2, "total_tokens": 2}
            })
        }

        async fn mock_tei_embed() -> MockServer {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/embeddings"))
                .respond_with(ResponseTemplate::new(200).set_body_json(openai_response()))
                .mount(&server)
                .await;
            server
        }

        #[tokio::test]
        async fn passthrough_preserves_body_on_both_spellings() {
            let tei = mock_tei_embed().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let body = serde_json::json!({"input": ["hi"], "model": "test-embed"});
            let (s1, j1) = post_json(&routes, "/v1/embeddings", &body).await;
            let (s2, j2) = post_json(&routes, "/embeddings", &body).await;

            assert_eq!((s1, s2), (200, 200));
            assert_eq!(j1, openai_response());
            assert_eq!(j2, openai_response());
        }

        #[tokio::test]
        async fn upstream_error_status_is_preserved_not_remapped() {
            let tei = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/embeddings"))
                .respond_with(ResponseTemplate::new(413).set_body_json(serde_json::json!({
                    "error": "batch size 2000 > maximum allowed batch size 1000"
                })))
                .mount(&tei)
                .await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let body = serde_json::json!({"input": ["hi"]});
            let (status, json) = post_json(&routes, "/v1/embeddings", &body).await;

            // Embeddings are a passthrough: TEI's own status survives (no 502 remap)
            assert_eq!(status, 413);
            assert!(json["error"].as_str().unwrap().contains("batch size"));
        }

        #[tokio::test]
        async fn non_object_body_is_400() {
            let tei = mock_tei_embed().await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let (status, json) =
                post_json(&routes, "/v1/embeddings", &serde_json::json!(["not", "object"])).await;

            assert_eq!(status, 400);
            assert_eq!(json["error"], "bad_request");
        }

        #[tokio::test]
        async fn unknown_fields_pass_through_untouched() {
            let tei = MockServer::start().await;
            // The proxy must not strip fields it doesn't know about
            Mock::given(method("POST"))
                .and(path("/v1/embeddings"))
                .and(body_json(serde_json::json!({
                    "input": ["hi"],
                    "model": "m",
                    "encoding_format": "float",
                    "custom_field": "untouched"
                })))
                .respond_with(ResponseTemplate::new(200).set_body_json(openai_response()))
                .mount(&tei)
                .await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let body = serde_json::json!({
                "input": ["hi"],
                "model": "m",
                "encoding_format": "float",
                "custom_field": "untouched"
            });
            let (status, _) = post_json(&routes, "/v1/embeddings", &body).await;

            assert_eq!(status, 200);
        }
    }

    mod health {
        use super::*;

        #[tokio::test]
        async fn healthy_when_both_upstreams_respond() {
            let tei = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/health"))
                .respond_with(ResponseTemplate::new(200))
                .mount(&tei)
                .await;
            let routes = build_routes(test_config(&tei.uri(), &tei.uri()), Client::new());

            let resp = warp::test::request()
                .method("GET")
                .path("/health")
                .reply(&routes)
                .await;
            let json: serde_json::Value = serde_json::from_slice(resp.body()).unwrap();

            assert_eq!(resp.status(), 200);
            assert_eq!(json["status"], "healthy");
            assert_eq!(json["upstreams"]["rerank"], true);
            assert_eq!(json["upstreams"]["embed"], true);
            assert!(json["endpoints"]["rerank"]["openwebui"].is_array());
        }

        #[tokio::test]
        async fn degraded_when_an_upstream_is_down() {
            let tei = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/health"))
                .respond_with(ResponseTemplate::new(200))
                .mount(&tei)
                .await;
            // Rerank upstream unreachable, embed healthy
            let routes = build_routes(
                test_config("http://127.0.0.1:9", &tei.uri()),
                Client::new(),
            );

            let resp = warp::test::request()
                .method("GET")
                .path("/health")
                .reply(&routes)
                .await;
            let json: serde_json::Value = serde_json::from_slice(resp.body()).unwrap();

            assert_eq!(resp.status(), 503);
            assert_eq!(json["status"], "degraded");
            assert_eq!(json["upstreams"]["rerank"], false);
            assert_eq!(json["upstreams"]["embed"], true);
        }
    }
}
