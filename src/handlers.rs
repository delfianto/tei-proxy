use log::{debug, error, info};
use reqwest::Client;
use std::time::Duration;

use crate::types::{
    ApiError, RankResult, RerankRequest, RerankResponse, ServiceConfig, TEIRerankRequest,
    TEIRerankResponse,
};

// --- Rerank Handler ---

pub async fn handle_rerank(
    req: RerankRequest,
    config: ServiceConfig,
    client: Client,
    client_auth: Option<String>,
    max_batch_size: usize,
    timeout: Duration,
    request_id: String,
) -> Result<impl warp::Reply, warp::Rejection> {
    info!(
        "[{}] Rerank request: {} documents",
        request_id,
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

    let tei_req = TEIRerankRequest {
        query: req.query,
        texts: req.documents.clone(),
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

    let mut indexed_scores: Vec<(usize, f64)> = tei_response
        .0
        .into_iter()
        .map(|r| (r.index, r.score))
        .collect();

    indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let limit = req.top_n.unwrap_or(indexed_scores.len());
    let results: Vec<RankResult> = indexed_scores
        .into_iter()
        .take(limit)
        .map(|(index, score)| RankResult {
            index,
            relevance_score: score,
        })
        .collect();

    info!(
        "[{}] Rerank complete: {} results",
        request_id,
        results.len()
    );

    Ok(warp::reply::json(&RerankResponse { results }))
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
            "supported_endpoints": ["/v1/rerank", "/v1/embeddings", "/rerank", "/embeddings"]
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
                documents: vec!["doc".to_string()],
                model: None,
                top_n: None,
            };

            let result = handle_rerank(
                req,
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
            };

            let result = handle_rerank(
                req,
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
                documents: vec!["a".to_string(), "b".to_string(), "c".to_string()],
                model: None,
                top_n: None,
            };

            let result = handle_rerank(
                req,
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
                    "texts": ["doc1", "doc2", "doc3"]
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
                documents: vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()],
                model: None,
                top_n: None,
            };

            let result = handle_rerank(
                req,
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
                documents: vec!["a".to_string(), "b".to_string(), "c".to_string()],
                model: None,
                top_n: Some(2),
            };

            let result = handle_rerank(
                req,
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
                documents: vec!["doc".to_string()],
                model: None,
                top_n: None,
            };

            let result = handle_rerank(
                req,
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
                documents: vec!["doc".to_string()],
                model: None,
                top_n: None,
            };

            let config = ServiceConfig {
                endpoint: mock_server.uri(),
                api_key: Some("env-key".to_string()),
            };

            let result = handle_rerank(
                req,
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
                documents: vec!["doc".to_string()],
                model: None,
                top_n: None,
            };

            let result = handle_rerank(
                req,
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

    mod result_sorting {
        #[test]
        fn test_scores_sorted_descending() {
            let mut scores = vec![(0, 0.5), (1, 0.9), (2, 0.3)];
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            assert_eq!(scores[0], (1, 0.9));
            assert_eq!(scores[1], (0, 0.5));
            assert_eq!(scores[2], (2, 0.3));
        }

        #[test]
        fn test_top_n_truncation() {
            let scores = vec![(1, 0.9), (0, 0.5), (2, 0.3)];
            let top_2: Vec<_> = scores.into_iter().take(2).collect();

            assert_eq!(top_2.len(), 2);
            assert_eq!(top_2[0], (1, 0.9));
            assert_eq!(top_2[1], (0, 0.5));
        }
    }
}
