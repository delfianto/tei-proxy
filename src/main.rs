use log::info;
use tei_proxy::routes::build_routes;
use tei_proxy::types::ProxyConfig;
use tokio::signal::unix::{SignalKind, signal};

#[tokio::main]
async fn main() {
    env_logger::init();

    let config = ProxyConfig::from_env();

    // Self-check mode for the Docker HEALTHCHECK: the runtime image is FROM
    // scratch (no shell, no curl), so the binary probes its own /health
    // endpoint — a deep check that includes both TEI upstreams — and exits
    // 0 (healthy) or 1 (degraded/unreachable).
    if std::env::args().any(|arg| arg == "--healthcheck") {
        let healthy = tei_proxy::handlers::self_check(config.port).await;
        std::process::exit(if healthy { 0 } else { 1 });
    }

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
    // Docker sends SIGTERM on stop, and as PID 1 in the scratch image an
    // unhandled signal is ignored outright (the kernel applies no default
    // action for init), so the container would ride out the full stop grace
    // period and get SIGKILLed. Listen for SIGTERM alongside Ctrl+C (SIGINT).
    let (tx, rx) = tokio::sync::oneshot::channel();

    let mut sigterm = signal(SignalKind::terminate()).expect("Failed to install SIGTERM handler");
    tokio::spawn(async move {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = sigterm.recv() => {},
        }
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
