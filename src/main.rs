mod handlers;
mod types;

use handlers::{handle_embedding, handle_health, handle_rerank};
use log::info;
use types::ProxyConfig;
use uuid::Uuid;
use warp::Filter;

#[tokio::main]
async fn main() {
    env_logger::init();

    let config = ProxyConfig::from_env();

    info!("Starting TEI proxy on port {}", config.port);
    info!(
        "Rerank: {} (auth: {})",
        config.rerank.endpoint,
        config.rerank.api_key.is_some()
    );
    info!(
        "Embed:  {} (auth: {})",
        config.embed.endpoint,
        config.embed.api_key.is_some()
    );
    info!("Max batch size: {}", config.max_batch_size);

    let http_client = reqwest::Client::builder()
        .pool_max_idle_per_host(10)
        .build()
        .expect("Failed to create HTTP client");

    // --- Contexts ---
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
    let health_client = reqwest::Client::new();

    let health = warp::path("health").and(warp::get()).and_then(move || {
        let r = health_rerank.clone();
        let e = health_embed.clone();
        let c = health_client.clone();
        async move { handle_health(r, e, c).await }
    });

    // --- Rerank Routes ---
    let rerank_handler = warp::post()
        .and(warp::body::json())
        .and(rerank_ctx.clone())
        .and(client_ctx.clone())
        .and(auth_header.clone())
        .and(max_batch_ctx)
        .and(rerank_timeout_ctx)
        .and(request_id.clone())
        .and_then(
            |req, config, client, auth, max_batch, timeout, req_id| async move {
                handle_rerank(req, config, client, auth, max_batch, timeout, req_id).await
            },
        );

    let v1_rerank = warp::path!("v1" / "rerank").and(rerank_handler.clone());
    let short_rerank = warp::path!("rerank").and(rerank_handler);

    // --- Embedding Routes ---
    let embed_handler = warp::post()
        .and(warp::body::json())
        .and(embed_ctx.clone())
        .and(client_ctx.clone())
        .and(auth_header.clone())
        .and(embed_timeout_ctx)
        .and(request_id.clone())
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

    // --- Compose Routes ---
    let routes = health
        .or(v1_rerank)
        .or(short_rerank)
        .or(v1_embed)
        .or(short_embed)
        .recover(types::handle_rejection)
        .with(cors)
        .with(warp::log("tei_proxy"));

    // --- Graceful Shutdown ---
    let (tx, rx) = tokio::sync::oneshot::channel();

    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for Ctrl+C");
        info!("Shutdown signal received");
        let _ = tx.send(());
    });

    let (_, server) =
        warp::serve(routes).bind_with_graceful_shutdown(([0, 0, 0, 0], config.port), async {
            rx.await.ok();
        });

    server.await;
    info!("Server stopped");
}
