use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;

// --- Configuration ---

#[derive(Clone, Debug)]
pub struct ServiceConfig {
    pub endpoint: String,
    pub api_key: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ProxyConfig {
    pub rerank: ServiceConfig,
    pub embed: ServiceConfig,
    pub port: u16,
    pub max_batch_size: usize,
    pub rerank_timeout: Duration,
    pub embed_timeout: Duration,
}

impl ProxyConfig {
    pub fn from_env() -> Self {
        let normalize_url = |url: String| -> String {
            if url.starts_with("http://") || url.starts_with("https://") {
                url
            } else {
                format!("http://{}", url)
            }
        };

        // Legacy fallback
        let default_tei =
            env::var("TEI_ENDPOINT").unwrap_or_else(|_| "http://localhost:4000".to_string());
        let default_tei = normalize_url(default_tei);

        let rerank_host = env::var("RERANKING_HOST").unwrap_or_else(|_| default_tei.clone());
        let rerank = ServiceConfig {
            endpoint: normalize_url(rerank_host),
            api_key: env::var("RERANKING_API_KEY").ok(),
        };

        let embed_host = env::var("EMBEDDING_HOST").unwrap_or_else(|_| default_tei.clone());
        let embed = ServiceConfig {
            endpoint: normalize_url(embed_host),
            api_key: env::var("EMBEDDING_API_KEY").ok(),
        };

        let port: u16 = env::var("TEI_PROXY_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8000);

        let max_batch_size: usize = env::var("MAX_CLIENT_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1000);

        let rerank_timeout = Duration::from_secs(
            env::var("RERANK_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(30),
        );

        let embed_timeout = Duration::from_secs(
            env::var("EMBED_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60),
        );

        Self {
            rerank,
            embed,
            port,
            max_batch_size,
            rerank_timeout,
            embed_timeout,
        }
    }
}

// --- Request/Response Types ---

/// OpenWebUI / OpenAI-style Rerank Request
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RerankRequest {
    pub query: String,
    pub documents: Vec<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub top_n: Option<usize>,
}

/// TEI Rerank Request (internal transform)
#[derive(Serialize, Debug, Clone)]
pub struct TEIRerankRequest {
    pub query: String,
    pub texts: Vec<String>,
}

/// TEI Rerank Response (internal)
#[derive(Deserialize, Debug)]
pub struct TEIRerankResponse(pub Vec<TEIRerankResult>);

#[derive(Deserialize, Debug)]
pub struct TEIRerankResult {
    pub index: usize,
    pub score: f64,
}

/// Public Rerank Response
#[derive(Serialize, Debug)]
pub struct RerankResponse {
    pub results: Vec<RankResult>,
}

#[derive(Serialize, Debug, PartialEq)]
pub struct RankResult {
    pub index: usize,
    pub relevance_score: f64,
}

#[derive(Serialize, Debug)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
}

// --- Error Handling ---

#[derive(Debug)]
pub enum ApiError {
    BadRequest(String),
    UpstreamError(String),
}

impl warp::reject::Reject for ApiError {}

pub async fn handle_rejection(
    err: warp::Rejection,
) -> Result<impl warp::Reply, std::convert::Infallible> {
    let (code, message, error_type) = if err.is_not_found() {
        (404, "Not Found".to_string(), "not_found")
    } else if let Some(api_error) = err.find::<ApiError>() {
        match api_error {
            ApiError::BadRequest(msg) => (400, msg.clone(), "bad_request"),
            ApiError::UpstreamError(msg) => (502, msg.clone(), "upstream_error"),
        }
    } else if err
        .find::<warp::filters::body::BodyDeserializeError>()
        .is_some()
    {
        (
            400,
            "Invalid JSON in request body".to_string(),
            "invalid_json",
        )
    } else {
        log::error!("Unhandled rejection: {:?}", err);
        (500, "Internal Server Error".to_string(), "internal_error")
    };

    let error_response = ErrorResponse {
        error: error_type.to_string(),
        message,
    };

    Ok(warp::reply::with_status(
        warp::reply::json(&error_response),
        warp::http::StatusCode::from_u16(code).unwrap(),
    ))
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    mod config {
        use super::*;
        use std::env;

        // Note: These tests manipulate global environment variables and may interfere
        // with each other when run in parallel. Run with `cargo test -- --test-threads=1`
        // if experiencing flakiness, or add the `serial_test` crate to run them serially.

        fn clear_env() {
            unsafe {
                env::remove_var("TEI_ENDPOINT");
                env::remove_var("RERANKING_HOST");
                env::remove_var("RERANKING_API_KEY");
                env::remove_var("EMBEDDING_HOST");
                env::remove_var("EMBEDDING_API_KEY");
                env::remove_var("TEI_PROXY_PORT");
                env::remove_var("MAX_CLIENT_BATCH_SIZE");
                env::remove_var("RERANK_TIMEOUT_SECS");
                env::remove_var("EMBED_TIMEOUT_SECS");
            }
        }

        #[test]
        fn test_defaults() {
            clear_env();
            let config = ProxyConfig::from_env();

            assert_eq!(config.port, 8000);
            assert_eq!(config.max_batch_size, 1000);
            assert_eq!(config.rerank_timeout, Duration::from_secs(30));
            assert_eq!(config.embed_timeout, Duration::from_secs(60));
            assert_eq!(config.rerank.endpoint, "http://localhost:4000");
            assert_eq!(config.embed.endpoint, "http://localhost:4000");
        }

        #[test]
        fn test_legacy_tei_endpoint_fallback() {
            clear_env();
            unsafe {
                env::set_var("TEI_ENDPOINT", "http://tei-server:5000");
            }
            let config = ProxyConfig::from_env();

            assert_eq!(config.rerank.endpoint, "http://tei-server:5000");
            assert_eq!(config.embed.endpoint, "http://tei-server:5000");
            clear_env();
        }

        #[test]
        fn test_specific_hosts_override_legacy() {
            clear_env();
            unsafe {
                env::set_var("TEI_ENDPOINT", "http://legacy:5000");
                env::set_var("RERANKING_HOST", "http://rerank:6000");
                env::set_var("EMBEDDING_HOST", "http://embed:7000");
            }
            let config = ProxyConfig::from_env();

            assert_eq!(config.rerank.endpoint, "http://rerank:6000");
            assert_eq!(config.embed.endpoint, "http://embed:7000");
            clear_env();
        }

        #[test]
        fn test_url_normalization_adds_scheme() {
            clear_env();
            unsafe {
                env::set_var("RERANKING_HOST", "rerank:6000");
            }
            let config = ProxyConfig::from_env();

            assert_eq!(config.rerank.endpoint, "http://rerank:6000");
            clear_env();
        }

        #[test]
        fn test_url_normalization_preserves_https() {
            clear_env();
            unsafe {
                env::set_var("RERANKING_HOST", "https://secure-rerank:6000");
            }
            let config = ProxyConfig::from_env();

            assert_eq!(config.rerank.endpoint, "https://secure-rerank:6000");
            clear_env();
        }

        #[test]
        fn test_custom_timeouts() {
            clear_env();
            unsafe {
                env::set_var("RERANK_TIMEOUT_SECS", "120");
                env::set_var("EMBED_TIMEOUT_SECS", "180");
            }
            let config = ProxyConfig::from_env();

            assert_eq!(config.rerank_timeout, Duration::from_secs(120));
            assert_eq!(config.embed_timeout, Duration::from_secs(180));
            clear_env();
        }

        #[test]
        fn test_invalid_port_falls_back_to_default() {
            clear_env();
            unsafe {
                env::set_var("TEI_PROXY_PORT", "not_a_number");
            }
            let config = ProxyConfig::from_env();

            assert_eq!(config.port, 8000);
            clear_env();
        }
    }

    mod serialization {
        use super::*;

        #[test]
        fn test_rerank_request_deserialize_minimal() {
            let json = r#"{"query": "test", "documents": ["a", "b"]}"#;
            let req: RerankRequest = serde_json::from_str(json).unwrap();

            assert_eq!(req.query, "test");
            assert_eq!(req.documents, vec!["a", "b"]);
            assert!(req.model.is_none());
            assert!(req.top_n.is_none());
        }

        #[test]
        fn test_rerank_request_deserialize_full() {
            let json = r#"{
                "query": "test",
                "documents": ["a", "b"],
                "model": "bge-reranker",
                "top_n": 5
            }"#;
            let req: RerankRequest = serde_json::from_str(json).unwrap();

            assert_eq!(req.model, Some("bge-reranker".to_string()));
            assert_eq!(req.top_n, Some(5));
        }

        #[test]
        fn test_tei_request_serialize() {
            let req = TEIRerankRequest {
                query: "search".to_string(),
                texts: vec!["doc1".to_string(), "doc2".to_string()],
            };
            let json = serde_json::to_string(&req).unwrap();

            assert!(json.contains(r#""query":"search""#));
            assert!(json.contains(r#""texts":["doc1","doc2"]"#));
        }

        #[test]
        fn test_rerank_response_serialize() {
            let resp = RerankResponse {
                results: vec![
                    RankResult {
                        index: 1,
                        relevance_score: 0.9,
                    },
                    RankResult {
                        index: 0,
                        relevance_score: 0.5,
                    },
                ],
            };
            let json = serde_json::to_string(&resp).unwrap();

            assert!(json.contains(r#""index":1"#));
            assert!(json.contains(r#""relevance_score":0.9"#));
        }

        #[test]
        fn test_error_response_serialize() {
            let resp = ErrorResponse {
                error: "bad_request".to_string(),
                message: "Query cannot be empty".to_string(),
            };
            let json = serde_json::to_string(&resp).unwrap();

            assert!(json.contains(r#""error":"bad_request""#));
            assert!(json.contains(r#""message":"Query cannot be empty""#));
        }
    }
}
