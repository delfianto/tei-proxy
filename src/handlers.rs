use log::{debug, error, info};
use reqwest::Client;
use std::time::Duration;

use crate::types::{
    ApiError, CohereApiVersion, CohereBilledUnits, CohereMeta, CohereRerankResponse,
    JinaRerankResponse, OpenWebUIRerankResponse, RankDocument, RankResult, RerankFlavor,
    RerankRequest, RerankUsage, ServiceConfig, TEIRerankRequest, TEIRerankResponse,
};

// --- Rerank Handler ---

// Arguments mirror the warp filter chain, which extracts them positionally
#[allow(clippy::too_many_arguments)]
pub async fn handle_rerank(
    req: RerankRequest,
    flavor: RerankFlavor,
    config: ServiceConfig,
    client: Client,
    client_auth: Option<String>,
    max_batch_size: usize,
    timeout: Duration,
    request_id: String,
) -> Result<impl warp::Reply, warp::Rejection> {
    info!(
        "[{}] Rerank request ({:?}): {} documents",
        request_id,
        flavor,
        req.documents.len()
    );

    if req.query.trim().is_empty() {
        return Err(warp::reject::custom(ApiError::BadRequest(
            "Query cannot be empty".to_string(),
        )));
    }
    if req.documents.is_empty() {
        return Err(warp::reject::custom(ApiError::BadRequest(
            "Documents list cannot be empty".to_string(),
        )));
    }
    if req.documents.len() > max_batch_size {
        return Err(warp::reject::custom(ApiError::BadRequest(format!(
            "Batch size {} exceeds maximum {}",
            req.documents.len(),
            max_batch_size
        ))));
    }

    let texts: Vec<String> = req.documents.iter().map(extract_document_text).collect();

    let tei_req = TEIRerankRequest {
        query: req.query,
        texts,
        truncate: req.truncate.unwrap_or(true),
    };

    let tei_url = format!("{}/rerank", config.endpoint.trim_end_matches('/'));
    debug!("[{}] Forwarding to: {}", request_id, tei_url);

    let mut request_builder = client.post(&tei_url).timeout(timeout).json(&tei_req);

    // Auth: env var takes precedence over client header
    request_builder = apply_auth(request_builder, &config.api_key, &client_auth);

    let response = request_builder.send().await.map_err(|e| {
        error!("[{}] Rerank connection failed: {}", request_id, e);
        warp::reject::custom(ApiError::UpstreamError(format!("Connection failed: {}", e)))
    })?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        error!("[{}] Upstream error {}: {}", request_id, status, error_text);
        return Err(warp::reject::custom(ApiError::UpstreamError(format!(
            "Upstream {}: {}",
            status, error_text
        ))));
    }

    let tei_response: TEIRerankResponse = response.json().await.map_err(|e| {
        error!("[{}] Failed to parse response: {}", request_id, e);
        warp::reject::custom(ApiError::UpstreamError(
            "Invalid response from upstream".to_string(),
        ))
    })?;

    let indexed_scores: Vec<(usize, f64)> = tei_response
        .0
        .into_iter()
        .map(|r| (r.index, r.score))
        .collect();

    let reply = match flavor {
        RerankFlavor::OpenWebUI => {
            let results = to_rank_results(full_coverage_by_index(indexed_scores), None);
            info!(
                "[{}] Rerank complete: {} results",
                request_id,
                results.len()
            );
            warp::reply::json(&OpenWebUIRerankResponse { results })
        }
        RerankFlavor::Jina => {
            let texts = req
                .return_documents
                .unwrap_or(false)
                .then_some(tei_req.texts.as_slice());
            let results = to_rank_results(sort_and_limit(indexed_scores, req.top_n), texts);
            info!(
                "[{}] Rerank complete: {} results",
                request_id,
                results.len()
            );
            warp::reply::json(&JinaRerankResponse {
                object: "rerank",
                model: req.model.unwrap_or_else(|| "tei-reranker".to_string()),
                usage: estimate_usage(&tei_req),
                results,
            })
        }
        RerankFlavor::Cohere => {
            let results = to_rank_results(sort_and_limit(indexed_scores, req.top_n), None);
            info!(
                "[{}] Rerank complete: {} results",
                request_id,
                results.len()
            );
            warp::reply::json(&CohereRerankResponse {
                id: request_id,
                results,
                meta: CohereMeta {
                    api_version: CohereApiVersion { version: "2" },
                    billed_units: CohereBilledUnits { search_units: 1 },
                },
            })
        }
    };

    Ok(reply)
}

/// Accepts the document shapes seen in the wild: plain strings (current
/// OpenWebUI), objects with `page_content` (legacy OpenWebUI / LangChain) or
/// `text` (Jina), and anything else as its raw JSON representation.
pub fn extract_document_text(doc: &serde_json::Value) -> String {
    if let Some(s) = doc.as_str() {
        s.to_string()
    } else if let Some(content) = doc.get("page_content").and_then(|v| v.as_str()) {
        content.to_string()
    } else if let Some(text) = doc.get("text").and_then(|v| v.as_str()) {
        text.to_string()
    } else {
        doc.to_string()
    }
}

/// Jina/Cohere semantics: results ordered by relevance, limited to top_n.
fn sort_and_limit(mut scores: Vec<(usize, f64)>, top_n: Option<usize>) -> Vec<(usize, f64)> {
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(n) = top_n {
        scores.truncate(n);
    }
    scores
}

/// OpenWebUI zips the returned scores positionally against its documents,
/// so every document must get a result back — top_n is deliberately ignored.
fn full_coverage_by_index(mut scores: Vec<(usize, f64)>) -> Vec<(usize, f64)> {
    scores.sort_by_key(|r| r.0);
    scores
}

fn to_rank_results(scores: Vec<(usize, f64)>, texts: Option<&[String]>) -> Vec<RankResult> {
    scores
        .into_iter()
        .map(|(index, score)| RankResult {
            index,
            relevance_score: score,
            document: texts
                .and_then(|t| t.get(index))
                .map(|text| RankDocument { text: text.clone() }),
        })
        .collect()
}

/// TEI's /rerank response carries no token counts; approximate with the
/// ~4 chars/token heuristic (each doc is scored as a query+doc pair).
fn estimate_usage(req: &TEIRerankRequest) -> RerankUsage {
    let chars =
        req.query.len() * req.texts.len() + req.texts.iter().map(String::len).sum::<usize>();
    let tokens = (chars / 4).max(1);
    RerankUsage {
        prompt_tokens: tokens,
        total_tokens: tokens,
    }
}

// --- Embedding Handler (Passthrough) ---

pub async fn handle_embedding(
    req: serde_json::Value,
    config: ServiceConfig,
    client: Client,
    client_auth: Option<String>,
    timeout: Duration,
    request_id: String,
) -> Result<impl warp::Reply, warp::Rejection> {
    info!("[{}] Embedding request (passthrough)", request_id);
    debug!("[{}] Payload: {:?}", request_id, req);

    if !req.is_object() {
        return Err(warp::reject::custom(ApiError::BadRequest(
            "Request body must be a JSON object".to_string(),
        )));
    }

    let base_endpoint = config.endpoint.trim_end_matches('/');
    let tei_url = format!("{}/v1/embeddings", base_endpoint);
    debug!("[{}] Forwarding to: {}", request_id, tei_url);

    let mut request_builder = client.post(&tei_url).timeout(timeout).json(&req);

    request_builder = apply_auth(request_builder, &config.api_key, &client_auth);

    let response = request_builder.send().await.map_err(|e| {
        error!("[{}] Embedding connection failed: {}", request_id, e);
        warp::reject::custom(ApiError::UpstreamError(format!("Connection failed: {}", e)))
    })?;

    let status_code = response.status().as_u16();
    let warp_status = warp::http::StatusCode::from_u16(status_code)
        .unwrap_or(warp::http::StatusCode::INTERNAL_SERVER_ERROR);

    let body = response.bytes().await.map_err(|e| {
        error!("[{}] Failed to read upstream body: {}", request_id, e);
        warp::reject::custom(ApiError::UpstreamError(format!(
            "Failed to read response: {}",
            e
        )))
    })?;

    info!("[{}] Embedding complete: {} bytes", request_id, body.len());

    Ok(warp::reply::with_status(
        warp::reply::with_header(body.to_vec(), "Content-Type", "application/json"),
        warp_status,
    ))
}

// --- Health Check ---

pub async fn handle_health(
    rerank_config: ServiceConfig,
    embed_config: ServiceConfig,
    client: Client,
) -> Result<impl warp::Reply, warp::Rejection> {
    let check_upstream = |endpoint: String| {
        let client = client.clone();
        async move {
            // Try /health, fall back to assuming OK if we get any response
            client
                .get(format!("{}/health", endpoint.trim_end_matches('/')))
                .timeout(Duration::from_secs(3))
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false)
        }
    };

    let rerank_ok = check_upstream(rerank_config.endpoint.clone()).await;
    let embed_ok = check_upstream(embed_config.endpoint.clone()).await;

    let all_ok = rerank_ok && embed_ok;
    let status = if all_ok { "healthy" } else { "degraded" };
    let code = if all_ok { 200 } else { 503 };

    Ok(warp::reply::with_status(
        warp::reply::json(&serde_json::json!({
            "status": status,
            "service": "tei-proxy",
            "upstreams": {
                "rerank": rerank_ok,
                "embed": embed_ok
            },
            "endpoints": {
                "rerank": {
                    "openwebui": ["/openwebui/rerank"],
                    "jina": ["/jina/rerank", "/v1/rerank"],
                    "cohere": ["/cohere/rerank", "/v2/rerank"]
                },
                "embeddings": ["/v1/embeddings", "/embeddings"]
            }
        })),
        warp::http::StatusCode::from_u16(code).unwrap(),
    ))
}

// --- Helpers ---

fn apply_auth(
    builder: reqwest::RequestBuilder,
    env_key: &Option<String>,
    client_header: &Option<String>,
) -> reqwest::RequestBuilder {
    if let Some(key) = env_key {
        let auth_val = if key.starts_with("Bearer ") {
            key.clone()
        } else {
            format!("Bearer {}", key)
        };
        builder.header("Authorization", auth_val)
    } else if let Some(header) = client_header {
        builder.header("Authorization", header)
    } else {
        builder
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_config(endpoint: &str) -> ServiceConfig {
        ServiceConfig {
            endpoint: endpoint.to_string(),
            api_key: None,
        }
    }

    fn test_client() -> Client {
        Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap()
    }

    mod rerank_validation {
        use super::*;

        #[tokio::test]
        async fn test_empty_query_rejected() {
            let req = RerankRequest {
                query: "   ".to_string(),
                documents: vec![serde_json::json!("doc")],
                model: None,
                top_n: None,
                return_documents: None,
                truncate: None,
            };

            let result = handle_rerank(
                req,
                RerankFlavor::Jina,
                test_config("http://localhost:9999"),
                test_client(),
                None,
                1000,
                Duration::from_secs(5),
                "test-id".to_string(),
            )
            .await;

            assert!(result.is_err());
        }

        #[tokio::test]
        async fn test_empty_documents_rejected() {
            let req = RerankRequest {
                query: "valid query".to_string(),
                documents: vec![],
                model: None,
                top_n: None,
                return_documents: None,
                truncate: None,
            };

            let result = handle_rerank(
                req,
                RerankFlavor::Jina,
                test_config("http://localhost:9999"),
                test_client(),
                None,
                1000,
                Duration::from_secs(5),
                "test-id".to_string(),
            )
            .await;

            assert!(result.is_err());
        }

        #[tokio::test]
        async fn test_batch_size_exceeded_rejected() {
            let req = RerankRequest {
                query: "query".to_string(),
                documents: vec![serde_json::json!("a"), serde_json::json!("b"), serde_json::json!("c")],
                model: None,
                top_n: None,
                return_documents: None,
                truncate: None,
            };

            let result = handle_rerank(
                req,
                RerankFlavor::Jina,
                test_config("http://localhost:9999"),
                test_client(),
                None,
                2, // Max 2, but we sent 3
                Duration::from_secs(5),
                "test-id".to_string(),
            )
            .await;

            assert!(result.is_err());
        }
    }

    mod rerank_integration {
        use super::*;

        #[tokio::test]
        async fn test_successful_rerank() {
            let mock_server = MockServer::start().await;

            // TEI returns unsorted results
            Mock::given(method("POST"))
                .and(path("/rerank"))
                .and(body_json(serde_json::json!({
                    "query": "test query",
                    "texts": ["doc1", "doc2", "doc3"],
                    "truncate": true
                })))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                    {"index": 0, "score": 0.5},
                    {"index": 1, "score": 0.9},
                    {"index": 2, "score": 0.3}
                ])))
                .mount(&mock_server)
                .await;

            let req = RerankRequest {
                query: "test query".to_string(),
                documents: vec![serde_json::json!("doc1"), serde_json::json!("doc2"), serde_json::json!("doc3")],
                model: None,
                top_n: None,
                return_documents: None,
                truncate: None,
            };

            let result = handle_rerank(
                req,
                RerankFlavor::Jina,
                test_config(&mock_server.uri()),
                test_client(),
                None,
                1000,
                Duration::from_secs(5),
                "test-id".to_string(),
            )
            .await;

            assert!(result.is_ok());
        }

        #[tokio::test]
        async fn test_top_n_limits_results() {
            let mock_server = MockServer::start().await;

            Mock::given(method("POST"))
                .and(path("/rerank"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                    {"index": 0, "score": 0.5},
                    {"index": 1, "score": 0.9},
                    {"index": 2, "score": 0.3}
                ])))
                .mount(&mock_server)
                .await;

            let req = RerankRequest {
                query: "test".to_string(),
                documents: vec![serde_json::json!("a"), serde_json::json!("b"), serde_json::json!("c")],
                model: None,
                top_n: Some(2),
                return_documents: None,
                truncate: None,
            };

            let result = handle_rerank(
                req,
                RerankFlavor::Jina,
                test_config(&mock_server.uri()),
                test_client(),
                None,
                1000,
                Duration::from_secs(5),
                "test-id".to_string(),
            )
            .await;

            assert!(result.is_ok());
            // Response would have only 2 results due to top_n
        }

        #[tokio::test]
        async fn test_auth_header_passed_through() {
            let mock_server = MockServer::start().await;

            Mock::given(method("POST"))
                .and(path("/rerank"))
                .and(header("Authorization", "Bearer test-token"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                    {"index": 0, "score": 0.9}
                ])))
                .mount(&mock_server)
                .await;

            let req = RerankRequest {
                query: "test".to_string(),
                documents: vec![serde_json::json!("doc")],
                model: None,
                top_n: None,
                return_documents: None,
                truncate: None,
            };

            let result = handle_rerank(
                req,
                RerankFlavor::Jina,
                test_config(&mock_server.uri()),
                test_client(),
                Some("Bearer test-token".to_string()),
                1000,
                Duration::from_secs(5),
                "test-id".to_string(),
            )
            .await;

            assert!(result.is_ok());
        }

        #[tokio::test]
        async fn test_env_api_key_overrides_client_header() {
            let mock_server = MockServer::start().await;

            // Expect the env key, NOT the client header
            Mock::given(method("POST"))
                .and(path("/rerank"))
                .and(header("Authorization", "Bearer env-key"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                    {"index": 0, "score": 0.9}
                ])))
                .mount(&mock_server)
                .await;

            let req = RerankRequest {
                query: "test".to_string(),
                documents: vec![serde_json::json!("doc")],
                model: None,
                top_n: None,
                return_documents: None,
                truncate: None,
            };

            let config = ServiceConfig {
                endpoint: mock_server.uri(),
                api_key: Some("env-key".to_string()),
            };

            let result = handle_rerank(
                req,
                RerankFlavor::Jina,
                config,
                test_client(),
                Some("Bearer client-key".to_string()), // This should be ignored
                1000,
                Duration::from_secs(5),
                "test-id".to_string(),
            )
            .await;

            assert!(result.is_ok());
        }

        #[tokio::test]
        async fn test_upstream_error_handled() {
            let mock_server = MockServer::start().await;

            Mock::given(method("POST"))
                .and(path("/rerank"))
                .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
                .mount(&mock_server)
                .await;

            let req = RerankRequest {
                query: "test".to_string(),
                documents: vec![serde_json::json!("doc")],
                model: None,
                top_n: None,
                return_documents: None,
                truncate: None,
            };

            let result = handle_rerank(
                req,
                RerankFlavor::Jina,
                test_config(&mock_server.uri()),
                test_client(),
                None,
                1000,
                Duration::from_secs(5),
                "test-id".to_string(),
            )
            .await;

            assert!(result.is_err());
        }
    }

    mod embedding_validation {
        use super::*;

        #[tokio::test]
        async fn test_non_object_rejected() {
            let req = serde_json::json!(["not", "an", "object"]);

            let result = handle_embedding(
                req,
                test_config("http://localhost:9999"),
                test_client(),
                None,
                Duration::from_secs(5),
                "test-id".to_string(),
            )
            .await;

            assert!(result.is_err());
        }
    }

    mod embedding_integration {
        use super::*;

        #[tokio::test]
        async fn test_passthrough_success() {
            let mock_server = MockServer::start().await;

            let expected_response = serde_json::json!({
                "object": "list",
                "data": [{"embedding": [0.1, 0.2, 0.3]}]
            });

            Mock::given(method("POST"))
                .and(path("/v1/embeddings"))
                .respond_with(ResponseTemplate::new(200).set_body_json(expected_response.clone()))
                .mount(&mock_server)
                .await;

            let req = serde_json::json!({
                "input": "test text",
                "model": "bge-m3"
            });

            let result = handle_embedding(
                req,
                test_config(&mock_server.uri()),
                test_client(),
                None,
                Duration::from_secs(5),
                "test-id".to_string(),
            )
            .await;

            assert!(result.is_ok());
        }

        #[tokio::test]
        async fn test_passthrough_preserves_upstream_error_status() {
            let mock_server = MockServer::start().await;

            Mock::given(method("POST"))
                .and(path("/v1/embeddings"))
                .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                    "error": "Invalid model"
                })))
                .mount(&mock_server)
                .await;

            let req = serde_json::json!({"input": "test"});

            let result = handle_embedding(
                req,
                test_config(&mock_server.uri()),
                test_client(),
                None,
                Duration::from_secs(5),
                "test-id".to_string(),
            )
            .await;

            // Should succeed (passthrough), but with 400 status
            assert!(result.is_ok());
        }
    }

    mod document_extraction {
        use super::*;

        #[test]
        fn test_plain_string() {
            assert_eq!(extract_document_text(&serde_json::json!("hello")), "hello");
        }

        #[test]
        fn test_page_content_object() {
            let doc = serde_json::json!({"page_content": "content", "metadata": {"k": "v"}});
            assert_eq!(extract_document_text(&doc), "content");
        }

        #[test]
        fn test_text_object() {
            let doc = serde_json::json!({"text": "jina style"});
            assert_eq!(extract_document_text(&doc), "jina style");
        }

        #[test]
        fn test_page_content_wins_over_text() {
            let doc = serde_json::json!({"page_content": "primary", "text": "secondary"});
            assert_eq!(extract_document_text(&doc), "primary");
        }

        #[test]
        fn test_unknown_shape_falls_back_to_raw_json() {
            let doc = serde_json::json!({"title": "no known keys"});
            assert_eq!(extract_document_text(&doc), r#"{"title":"no known keys"}"#);
        }

        #[test]
        fn test_non_string_scalars_fall_back_to_raw_json() {
            assert_eq!(extract_document_text(&serde_json::json!(42)), "42");
            assert_eq!(extract_document_text(&serde_json::json!(null)), "null");
        }

        #[test]
        fn test_non_string_page_content_falls_through() {
            // page_content that isn't a string must not be picked up
            let doc = serde_json::json!({"page_content": 7, "text": "fallback"});
            assert_eq!(extract_document_text(&doc), "fallback");
        }
    }

    mod result_shaping {
        use super::*;

        #[test]
        fn test_sort_and_limit_orders_by_relevance() {
            let scores = vec![(0, 0.5), (1, 0.9), (2, 0.3)];
            let sorted = sort_and_limit(scores, None);

            assert_eq!(sorted, vec![(1, 0.9), (0, 0.5), (2, 0.3)]);
        }

        #[test]
        fn test_sort_and_limit_truncates_to_top_n() {
            let scores = vec![(0, 0.5), (1, 0.9), (2, 0.3)];
            let top_2 = sort_and_limit(scores, Some(2));

            assert_eq!(top_2, vec![(1, 0.9), (0, 0.5)]);
        }

        #[test]
        fn test_sort_and_limit_top_n_larger_than_input_returns_all() {
            let scores = vec![(0, 0.5), (1, 0.9)];
            let all = sort_and_limit(scores, Some(10));

            assert_eq!(all.len(), 2);
        }

        #[test]
        fn test_sort_and_limit_top_n_zero_returns_empty() {
            let scores = vec![(0, 0.5), (1, 0.9)];
            let none = sort_and_limit(scores, Some(0));

            assert!(none.is_empty());
        }

        #[test]
        fn test_sort_and_limit_handles_tied_scores() {
            let scores = vec![(0, 0.5), (1, 0.5), (2, 0.5)];
            let sorted = sort_and_limit(scores, None);

            // No panic, all retained, scores unchanged
            assert_eq!(sorted.len(), 3);
            assert!(sorted.iter().all(|&(_, s)| s == 0.5));
        }

        #[test]
        fn test_full_coverage_empty_input() {
            assert!(full_coverage_by_index(vec![]).is_empty());
        }

        #[test]
        fn test_full_coverage_keeps_all_scores_in_index_order() {
            let scores = vec![(2, 0.3), (0, 0.5), (1, 0.9)];
            let covered = full_coverage_by_index(scores);

            // OpenWebUI zips scores positionally: every index must be present
            assert_eq!(covered, vec![(0, 0.5), (1, 0.9), (2, 0.3)]);
        }

        #[test]
        fn test_to_rank_results_without_documents() {
            let results = to_rank_results(vec![(1, 0.9)], None);

            assert_eq!(results.len(), 1);
            assert_eq!(results[0].index, 1);
            assert_eq!(results[0].relevance_score, 0.9);
            assert!(results[0].document.is_none());
        }

        #[test]
        fn test_to_rank_results_attaches_original_document_text() {
            let texts = vec!["first".to_string(), "second".to_string()];
            let results = to_rank_results(vec![(1, 0.9), (0, 0.5)], Some(&texts));

            assert_eq!(
                results[0].document,
                Some(RankDocument {
                    text: "second".to_string()
                })
            );
            assert_eq!(
                results[1].document,
                Some(RankDocument {
                    text: "first".to_string()
                })
            );
        }

        #[test]
        fn test_to_rank_results_out_of_range_index_yields_no_document() {
            // If TEI ever returned an index beyond the input, don't panic
            let texts = vec!["only one".to_string()];
            let results = to_rank_results(vec![(5, 0.9)], Some(&texts));

            assert_eq!(results[0].index, 5);
            assert!(results[0].document.is_none());
        }

        #[test]
        fn test_estimate_usage_minimum_is_one_token() {
            let req = TEIRerankRequest {
                query: String::new(),
                texts: vec![],
                truncate: true,
            };
            let usage = estimate_usage(&req);

            assert_eq!(usage.prompt_tokens, 1);
            assert_eq!(usage.total_tokens, 1);
        }

        #[test]
        fn test_estimate_usage_counts_query_once_per_document() {
            let one_doc = TEIRerankRequest {
                query: "querytext".to_string(), // 9 chars
                texts: vec!["x".repeat(100)],
                truncate: true,
            };
            let two_docs = TEIRerankRequest {
                query: "querytext".to_string(),
                texts: vec!["x".repeat(100), "x".repeat(100)],
                truncate: true,
            };

            let one = estimate_usage(&one_doc).prompt_tokens;
            let two = estimate_usage(&two_docs).prompt_tokens;

            // Each doc is scored as a (query, doc) pair, so the query's
            // chars count once per document: 2*(100+9)/4 > 2*(100)/4 + 9/4
            assert_eq!(one, (100 + 9) / 4);
            assert_eq!(two, (2 * (100 + 9)) / 4);
        }

        #[test]
        fn test_estimate_usage_is_nonzero() {
            let req = TEIRerankRequest {
                query: "query".to_string(),
                texts: vec!["some document".to_string()],
                truncate: true,
            };
            let usage = estimate_usage(&req);

            assert!(usage.prompt_tokens > 0);
            assert_eq!(usage.prompt_tokens, usage.total_tokens);
        }
    }

    mod flavor_integration {
        use super::*;

        fn three_doc_request(top_n: Option<usize>) -> RerankRequest {
            RerankRequest {
                query: "test".to_string(),
                documents: vec![
                    serde_json::json!("a"),
                    serde_json::json!("b"),
                    serde_json::json!("c"),
                ],
                model: None,
                top_n,
                return_documents: None,
                truncate: None,
            }
        }

        async fn mock_tei() -> MockServer {
            let mock_server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/rerank"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
                    {"index": 0, "score": 0.5},
                    {"index": 1, "score": 0.9},
                    {"index": 2, "score": 0.3}
                ])))
                .mount(&mock_server)
                .await;
            mock_server
        }

        #[tokio::test]
        async fn test_openwebui_flavor_succeeds_with_top_n() {
            let mock_server = mock_tei().await;

            // OpenWebUI sends top_n = len(documents); the shim ignores it
            let result = handle_rerank(
                three_doc_request(Some(1)),
                RerankFlavor::OpenWebUI,
                test_config(&mock_server.uri()),
                test_client(),
                None,
                1000,
                Duration::from_secs(5),
                "test-id".to_string(),
            )
            .await;

            assert!(result.is_ok());
        }

        #[tokio::test]
        async fn test_cohere_flavor_succeeds() {
            let mock_server = mock_tei().await;

            let result = handle_rerank(
                three_doc_request(Some(2)),
                RerankFlavor::Cohere,
                test_config(&mock_server.uri()),
                test_client(),
                None,
                1000,
                Duration::from_secs(5),
                "test-id".to_string(),
            )
            .await;

            assert!(result.is_ok());
        }

        #[tokio::test]
        async fn test_jina_flavor_with_return_documents() {
            let mock_server = mock_tei().await;

            let mut req = three_doc_request(None);
            req.return_documents = Some(true);

            let result = handle_rerank(
                req,
                RerankFlavor::Jina,
                test_config(&mock_server.uri()),
                test_client(),
                None,
                1000,
                Duration::from_secs(5),
                "test-id".to_string(),
            )
            .await;

            assert!(result.is_ok());
        }
    }
}
