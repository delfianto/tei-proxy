use log::{debug, error, info};
use serde::{Deserialize, Serialize};
use std::env;
use warp::Filter;

// --- Data Structures ---

#[derive(Clone, Debug)]
struct ServiceConfig {
    endpoint: String,
    api_key: Option<String>,
}

// OpenWebUI / OpenAI-style Rerank Request
#[derive(Serialize, Deserialize, Debug, Clone)]
struct RerankRequest {
    query: String,
    documents: Vec<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    top_n: Option<usize>,
}

// TEI Rerank Request (Internal)
#[derive(Serialize, Debug)]
struct TEIRerankRequest {
    query: String,
    texts: Vec<String>,
}

// TEI Rerank Response (Internal)
#[derive(Deserialize, Debug)]
struct TEIRerankResponse(Vec<TEIRerankResult>);

#[derive(Deserialize, Debug)]
struct TEIRerankResult {
    index: usize,
    score: f64,
}

// Public Rerank Response
#[derive(Serialize, Debug)]
struct RerankResponse {
    results: Vec<RankResult>,
}

#[derive(Serialize, Debug)]
struct RankResult {
    index: usize,
    relevance_score: f64,
}

#[derive(Serialize, Debug)]
struct ErrorResponse {
    error: String,
    message: String,
}

#[tokio::main]
async fn main() {
    // Initialize logger
    env_logger::init();

    // --- Configuration Loading ---

    // Helper to ensure URL has scheme
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

    // Reranking Config
    let rerank_host_raw = env::var("RERANKING_HOST").unwrap_or_else(|_| default_tei.clone());
    let rerank_key = env::var("RERANKING_API_KEY").ok();
    let rerank_config = ServiceConfig {
        endpoint: normalize_url(rerank_host_raw),
        api_key: rerank_key,
    };

    // Embedding Config
    let embed_host_raw = env::var("EMBEDDING_HOST").unwrap_or_else(|_| default_tei.clone());
    let embed_key = env::var("EMBEDDING_API_KEY").ok();
    let embed_config = ServiceConfig {
        endpoint: normalize_url(embed_host_raw),
        api_key: embed_key,
    };

    let port: u16 = env::var("TEI_PROXY_PORT")
        .unwrap_or_else(|_| "8000".to_string())
        .parse()
        .unwrap_or(8000);

    info!("Starting AI proxy server on port {}", port);
    info!(
        "Reranking Host: {} (Key present: {})",
        rerank_config.endpoint,
        rerank_config.api_key.is_some()
    );
    info!(
        "Embedding Host: {} (Key present: {})",
        embed_config.endpoint,
        embed_config.api_key.is_some()
    );

    // --- Filters ---

    let rerank_ctx = warp::any().map(move || rerank_config.clone());
    let embed_ctx = warp::any().map(move || embed_config.clone());

    // Extract Authorization header if present
    let auth_header = warp::header::optional::<String>("authorization");

    // Health check
    let health = warp::path("health").and(warp::get()).map(|| {
        warp::reply::json(&serde_json::json!({
            "status": "healthy",
            "service": "tei-proxy",
            "supported_endpoints": ["/v1/rerank", "/v1/embeddings", "/rerank", "/embed"]
        }))
    });

    // --- RERANK ROUTES ---
    let rerank_handler = warp::post()
        .and(warp::body::json())
        .and(rerank_ctx.clone())
        .and(auth_header.clone())
        .and_then(handle_rerank);

    let v1_rerank = warp::path("v1")
        .and(warp::path("rerank"))
        .and(rerank_handler.clone());
    let short_rerank = warp::path("rerank").and(rerank_handler.clone());

    // --- EMBEDDING ROUTES ---
    // Accept ANY valid JSON body for passthrough
    let embed_handler = warp::post()
        .and(warp::body::json())
        .and(embed_ctx.clone())
        .and(auth_header.clone())
        .and_then(handle_embedding_passthrough);

    let v1_embeddings = warp::path("v1")
        .and(warp::path("embeddings"))
        .and(embed_handler.clone());
    let short_embed = warp::path("embeddings").and(embed_handler.clone());

    // CORS
    let cors = warp::cors()
        .allow_any_origin()
        .allow_headers(vec!["content-type", "authorization"])
        .allow_methods(vec!["GET", "POST", "OPTIONS"]);

    let routes = health
        .or(v1_rerank)
        .or(short_rerank)
        .or(v1_embeddings)
        .or(short_embed)
        .recover(handle_rejection)
        .with(cors)
        .with(warp::log("tei_proxy"));

    warp::serve(routes).run(([0, 0, 0, 0], port)).await;
}

// --- HANDLERS ---

async fn handle_rerank(
    req: RerankRequest,
    config: ServiceConfig,
    client_auth: Option<String>,
) -> Result<impl warp::Reply, warp::Rejection> {
    info!("🔄 Rerank Request: {} docs", req.documents.len());

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

    let tei_req = TEIRerankRequest {
        query: req.query,
        texts: req.documents.clone(),
    };

    let client_builder = reqwest::Client::builder().timeout(std::time::Duration::from_secs(30));

    let client = client_builder
        .build()
        .map_err(|e| warp::reject::custom(ApiError::InternalError(e.to_string())))?;

    let tei_url = format!("{}/rerank", config.endpoint);

    let mut request_builder = client.post(&tei_url).json(&tei_req);

    // Auth Priority: Env Var > Client Header
    if let Some(key) = &config.api_key {
        let auth_val = if key.starts_with("Bearer ") {
            key.clone()
        } else {
            format!("Bearer {}", key)
        };
        request_builder = request_builder.header("Authorization", auth_val);
    } else if let Some(auth_header) = client_auth {
        request_builder = request_builder.header("Authorization", auth_header);
    }

    let response = request_builder.send().await.map_err(|e| {
        error!("Rerank connection failed: {}", e);
        warp::reject::custom(ApiError::TEIError(e.to_string()))
    })?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown TEI error".to_string());
        error!("Rerank upstream error {}: {}", status, error_text);
        return Err(warp::reject::custom(ApiError::TEIError(format!(
            "Upstream {}: {}",
            status, error_text
        ))));
    }

    let tei_response: TEIRerankResponse = response.json().await.map_err(|e| {
        error!("Failed to parse Rerank response: {}", e);
        warp::reject::custom(ApiError::TEIError(
            "Invalid response from Upstream".to_string(),
        ))
    })?;

    // Sort
    let mut indexed_scores: Vec<(usize, f64)> = tei_response
        .0
        .into_iter()
        .map(|result| (result.index, result.score))
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

    Ok(warp::reply::json(&RerankResponse { results }))
}

// Passthrough Handler for Embeddings
async fn handle_embedding_passthrough(
    req: serde_json::Value,
    config: ServiceConfig,
    client_auth: Option<String>,
) -> Result<impl warp::Reply, warp::Rejection> {
    info!("🧠 Embed Request (Passthrough)");
    debug!("Payload: {:?}", req);

    let base_endpoint = config.endpoint.trim_end_matches('/');
    // Assumes TEI endpoint supports /v1/embeddings or is OpenAI compatible
    let tei_url = format!("{}/v1/embeddings", base_endpoint);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| warp::reject::custom(ApiError::InternalError(e.to_string())))?;

    let mut request_builder = client.post(&tei_url).json(&req);

    if let Some(key) = &config.api_key {
        let auth_val = if key.starts_with("Bearer ") {
            key.clone()
        } else {
            format!("Bearer {}", key)
        };
        request_builder = request_builder.header("Authorization", auth_val);
    } else if let Some(auth_header) = client_auth {
        request_builder = request_builder.header("Authorization", auth_header);
    }

    let response = request_builder.send().await.map_err(|e| {
        error!("Embed connection failed: {}", e);
        warp::reject::custom(ApiError::TEIError(e.to_string()))
    })?;

    // Fix 1: Convert reqwest StatusCode to warp::http::StatusCode using u16
    let status_code_u16 = response.status().as_u16();
    let warp_status = warp::http::StatusCode::from_u16(status_code_u16)
        .unwrap_or(warp::http::StatusCode::INTERNAL_SERVER_ERROR);

    let body = response.bytes().await.map_err(|e| {
        error!("Failed to read upstream body: {}", e);
        warp::reject::custom(ApiError::TEIError(e.to_string()))
    })?;

    // Fix 2: Convert Bytes to Vec<u8> because Bytes doesn't implement warp::Reply
    let body_vec = body.to_vec();

    Ok(warp::reply::with_status(
        warp::reply::with_header(body_vec, "Content-Type", "application/json"),
        warp_status,
    ))
}

// --- ERROR HANDLING ---

#[derive(Debug)]
enum ApiError {
    BadRequest(String),
    TEIError(String),
    InternalError(String),
}

impl warp::reject::Reject for ApiError {}

async fn handle_rejection(
    err: warp::Rejection,
) -> Result<impl warp::Reply, std::convert::Infallible> {
    let (code, message, error_type) = if err.is_not_found() {
        (404, "Not Found".to_string(), "not_found")
    } else if let Some(api_error) = err.find::<ApiError>() {
        match api_error {
            ApiError::BadRequest(msg) => (400, msg.clone(), "bad_request"),
            ApiError::TEIError(msg) => (502, msg.clone(), "tei_error"),
            ApiError::InternalError(msg) => (500, msg.clone(), "internal_error"),
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
        error!("Unhandled rejection: {:?}", err);
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
