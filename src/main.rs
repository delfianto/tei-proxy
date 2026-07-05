use log::info;
use tei_proxy::routes::build_routes;
use tei_proxy::types::ProxyConfig;

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

    let routes = build_routes(config.clone(), http_client);

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
