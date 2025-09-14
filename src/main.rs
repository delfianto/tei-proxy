use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use std::env;
use warp::Filter;

#[derive(Serialize, Deserialize, Debug)]
struct OpenWebUIRequest {
    query: String,
    documents: Vec<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    top_n: Option<usize>,
}

#[derive(Serialize, Debug)]
struct TEIRequest {
    query: String,
    texts: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct TEIResponse(Vec<TEIRankResult>);

#[derive(Deserialize, Debug)]
struct TEIRankResult {
    index: usize,
    score: f64,
}

#[derive(Serialize, Debug)]
struct OpenWebUIResponse {
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

    // Get configuration from environment
    let tei_endpoint =
        env::var("TEI_ENDPOINT").unwrap_or_else(|_| "http://localhost:4000".to_string());

    let port: u16 = env::var("TEI_PROXY_PORT")
        .unwrap_or_else(|_| "8000".to_string())
        .parse()
        .unwrap_or(8000);

    info!("Starting rerank proxy server");
    info!("TEI endpoint: {}", tei_endpoint);
    info!("Listening on port: {}", port);

    // Health check endpoint
    let health = warp::path("health").and(warp::get()).map(|| {
        warp::reply::json(&serde_json::json!({
            "status": "healthy",
            "service": "rerank-proxy"
        }))
    });

    // Rerank endpoint with error handling
    let rerank = warp::path("rerank")
        .and(warp::post())
        .and(warp::body::json())
        .and(warp::any().map(move || tei_endpoint.clone()))
        .and_then(handle_rerank)
        .recover(handle_rejection);

    // CORS support
    let cors = warp::cors()
        .allow_any_origin()
        .allow_headers(vec!["content-type", "authorization"])
        .allow_methods(vec!["GET", "POST", "OPTIONS"]);

    let routes = health.or(rerank).with(cors).with(warp::log("rerank_proxy"));

    info!("Server started successfully");
    warp::serve(routes).run(([0, 0, 0, 0], port)).await;
}

async fn handle_rerank(
    req: OpenWebUIRequest,
    tei_endpoint: String,
) -> Result<impl warp::Reply, warp::Rejection> {
    info!("🔄 Processing rerank request for query: '{}'", req.query);
    info!(
        "📊 Number of documents: {}, top_n: {:?}",
        req.documents.len(),
        req.top_n
    );

    // Debug: Log the complete incoming request from WebUI
    match serde_json::to_string_pretty(&req) {
        Ok(json_str) => debug!("📥 Complete WebUI Request:\n{}", json_str),
        Err(e) => warn!("❌ Failed to serialize WebUI request for debug: {}", e),
    }

    // Validate input
    if req.query.trim().is_empty() {
        warn!("Empty query received");
        return Err(warp::reject::custom(ApiError::BadRequest(
            "Query cannot be empty".to_string(),
        )));
    }

    if req.documents.is_empty() {
        warn!("No documents provided");
        return Err(warp::reject::custom(ApiError::BadRequest(
            "Documents list cannot be empty".to_string(),
        )));
    }

    let max_batch_size = env::var("MAX_CLIENT_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| 1000);

    if req.documents.len() > max_batch_size {
        warn!("Too many documents: {}", req.documents.len());
        return Err(warp::reject::custom(ApiError::BadRequest(
            format!("Too many documents, max: {}", max_batch_size).to_string(),
        )));
    }

    // Transform to TEI format
    let tei_req = TEIRequest {
        query: req.query.clone(),
        texts: req.documents.clone(),
    };

    // Debug: Log the request being sent to TEI
    match serde_json::to_string_pretty(&tei_req) {
        Ok(json_str) => debug!("📤 TEI Request:\n{}", json_str),
        Err(e) => warn!("❌ Failed to serialize TEI request for debug: {}", e),
    }

    info!("🚀 Forwarding request to TEI endpoint: {}", tei_endpoint);

    // Call TEI endpoint with timeout and retries
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| {
            error!("Failed to create HTTP client: {}", e);
            warp::reject::custom(ApiError::InternalError(
                "HTTP client creation failed".to_string(),
            ))
        })?;

    let tei_url = format!("{}/rerank", tei_endpoint);
    let response = client
        .post(&tei_url)
        .json(&tei_req)
        .send()
        .await
        .map_err(|e| {
            error!("TEI request failed: {}", e);
            warp::reject::custom(ApiError::TEIError(format!(
                "Failed to connect to TEI service: {}",
                e
            )))
        })?;

    // Check response status
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        error!("TEI returned error {}: {}", status, error_text);
        return Err(warp::reject::custom(ApiError::TEIError(format!(
            "TEI service error {}: {}",
            status, error_text
        ))));
    }

    // Get response text first for debugging
    let response_text = response.text().await.map_err(|e| {
        error!("Failed to read TEI response body: {}", e);
        warp::reject::custom(ApiError::TEIError(
            "Failed to read response from TEI service".to_string(),
        ))
    })?;

    // Debug: Log the complete TEI response with pretty formatting
    match serde_json::from_str::<serde_json::Value>(&response_text) {
        Ok(json_value) => {
            let pretty_json =
                serde_json::to_string_pretty(&json_value).unwrap_or_else(|_| response_text.clone());
            debug!("📨 TEI Response:\n{}", pretty_json);
        }
        Err(_) => {
            debug!("📨 TEI Response (raw text):\n{}", response_text);
        }
    }

    // Parse TEI response
    let tei_response: TEIResponse = serde_json::from_str(&response_text).map_err(|e| {
        error!(
            "Failed to parse TEI response: {}. Raw response: {}",
            e, response_text
        );
        warp::reject::custom(ApiError::TEIError(format!(
            "Invalid response format from TEI service. Expected array of scores, got: {}",
            response_text
        )))
    })?;

    // Validate TEI response
    if tei_response.0.len() != req.documents.len() {
        error!(
            "TEI response length mismatch: expected {}, got {}",
            req.documents.len(),
            tei_response.0.len()
        );
        return Err(warp::reject::custom(ApiError::TEIError(
            "TEI response length doesn't match input documents".to_string(),
        )));
    }

    info!(
        "✅ TEI request successful, processing {} scores",
        tei_response.0.len()
    );

    // Transform back to OpenWebUI format with ranking
    // TEI returns results with indices, but we need to sort by score
    let mut indexed_scores: Vec<(usize, f64)> = tei_response
        .0
        .into_iter()
        .map(|result| (result.index, result.score))
        .collect();

    // Sort by relevance score descending
    indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let results: Vec<RankResult> = indexed_scores
        .into_iter()
        .map(|(index, score)| RankResult {
            index,
            relevance_score: score,
        })
        .collect();

    let response = OpenWebUIResponse { results };

    // Debug: Log the final response being sent back to WebUI
    match serde_json::to_string_pretty(&response) {
        Ok(json_str) => debug!("📤 Final WebUI Response:\n{}", json_str),
        Err(e) => warn!("❌ Failed to serialize WebUI response for debug: {}", e),
    }

    info!(
        "✅ Successfully processed rerank request, returning {} results",
        response.results.len()
    );
    Ok(warp::reply::json(&response))
}

// Custom error types
#[derive(Debug)]
enum ApiError {
    BadRequest(String),
    TEIError(String),
    InternalError(String),
}

impl warp::reject::Reject for ApiError {}

// Error handling
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
