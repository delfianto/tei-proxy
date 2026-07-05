//! Live integration tests against real HuggingFace TEI instances.
//!
//! Expects TEI running with an **embedding model at localhost:4001** and a
//! **reranker at localhost:4002** (override via `LIVE_TEI_EMBED` /
//! `LIVE_TEI_RERANK`). Every test first probes both upstreams and fails with
//! a clear message if either is unreachable.
//!
//! Marked `#[ignore]` so plain `cargo test` stays green without the servers:
//!
//! ```bash
//! cargo test --test live_tei -- --ignored
//! ```

use std::time::Duration;

use tei_proxy::routes::build_routes;
use tei_proxy::types::{ProxyConfig, ServiceConfig};

fn embed_url() -> String {
    std::env::var("LIVE_TEI_EMBED").unwrap_or_else(|_| "http://localhost:4001".to_string())
}

fn rerank_url() -> String {
    std::env::var("LIVE_TEI_RERANK").unwrap_or_else(|_| "http://localhost:4002".to_string())
}

fn live_config() -> ProxyConfig {
    ProxyConfig {
        rerank: ServiceConfig {
            endpoint: rerank_url(),
            api_key: None,
        },
        embed: ServiceConfig {
            endpoint: embed_url(),
            api_key: None,
        },
        port: 0,
        max_batch_size: 1000,
        rerank_timeout: Duration::from_secs(30),
        embed_timeout: Duration::from_secs(60),
    }
}

async fn probe(base: &str) -> Result<(), String> {
    let client = reqwest::Client::new();
    match client
        .get(format!("{}/health", base.trim_end_matches('/')))
        .timeout(Duration::from_secs(3))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => Ok(()),
        Ok(r) => Err(format!("{base}/health returned HTTP {}", r.status())),
        Err(e) => Err(format!("{base} is unreachable: {e}")),
    }
}

/// Fail fast with a readable message when the live upstreams aren't running.
async fn require_live() {
    let mut problems = Vec::new();
    if let Err(e) = probe(&embed_url()).await {
        problems.push(format!("embedding TEI: {e}"));
    }
    if let Err(e) = probe(&rerank_url()).await {
        problems.push(format!("rerank TEI: {e}"));
    }
    assert!(
        problems.is_empty(),
        "live TEI upstreams not available:\n  {}\nStart TEI (embedding at :4001, reranker at :4002) or run without --ignored.",
        problems.join("\n  ")
    );
}

async fn post_json(
    routes: &(impl warp::Filter<Extract = (impl warp::Reply + 'static,), Error = warp::Rejection>
          + Clone
          + 'static),
    path: &str,
    body: &serde_json::Value,
) -> (u16, serde_json::Value) {
    let resp = warp::test::request()
        .method("POST")
        .path(path)
        .header("content-type", "application/json")
        .json(body)
        .reply(routes)
        .await;
    let status = resp.status().as_u16();
    let json = serde_json::from_slice(resp.body()).unwrap_or(serde_json::Value::Null);
    (status, json)
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn embedding_vec(data: &serde_json::Value) -> Vec<f64> {
    data["embedding"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

const RELEVANT_DOC: &str =
    "Deep Learning is a subfield of machine learning based on artificial neural networks.";
const CHEESE_DOC: &str = "Cheese is a dairy product made from the milk of cows, goats, or sheep.";
const PARIS_DOC: &str = "The capital of France is Paris, a city on the river Seine.";
const QUERY: &str = "What is Deep Learning?";

// --- Reachability probes (the canary tests: run these first when debugging) ---

#[tokio::test]
#[ignore = "requires live TEI at :4001/:4002"]
async fn upstreams_reachable() {
    require_live().await;
}

#[tokio::test]
#[ignore = "requires live TEI at :4001/:4002"]
async fn upstreams_serve_the_expected_model_types() {
    require_live().await;
    let client = reqwest::Client::new();

    // Guard against the two ports being swapped in the environment
    let embed_info: serde_json::Value = client
        .get(format!("{}/info", embed_url()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        embed_info["model_type"].get("embedding").is_some(),
        "{} does not serve an embedding model: model_type = {}",
        embed_url(),
        embed_info["model_type"]
    );

    let rerank_info: serde_json::Value = client
        .get(format!("{}/info", rerank_url()))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        rerank_info["model_type"].get("reranker").is_some()
            || rerank_info["model_type"].get("classifier").is_some(),
        "{} does not serve a reranker: model_type = {}",
        rerank_url(),
        rerank_info["model_type"]
    );
}

// --- Rerank through the real model ---

#[tokio::test]
#[ignore = "requires live TEI at :4001/:4002"]
async fn live_openwebui_rerank_scores_every_document() {
    require_live().await;
    let routes = build_routes(live_config(), reqwest::Client::new());

    // top_n = 1 on purpose: the shim must still score everything
    let body = serde_json::json!({
        "model": "reranker",
        "query": QUERY,
        "documents": [RELEVANT_DOC, CHEESE_DOC, PARIS_DOC],
        "top_n": 1
    });
    let (status, json) = post_json(&routes, "/openwebui/rerank", &body).await;

    assert_eq!(status, 200, "body: {json}");
    let results = json["results"].as_array().unwrap();
    assert_eq!(results.len(), 3, "one result per document, always");

    // Index-ordered, sigmoid-normalized scores
    let scores: Vec<f64> = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            assert_eq!(r["index"].as_u64().unwrap() as usize, i);
            let s = r["relevance_score"].as_f64().unwrap();
            assert!((0.0..=1.0).contains(&s), "expected sigmoid score, got {s}");
            s
        })
        .collect();

    // Semantic sanity: the deep-learning doc must beat cheese and Paris
    assert!(
        scores[0] > scores[1] && scores[0] > scores[2],
        "relevant doc should score highest: {scores:?}"
    );
}

#[tokio::test]
#[ignore = "requires live TEI at :4001/:4002"]
async fn live_jina_rerank_sorts_limits_and_returns_documents() {
    require_live().await;
    let routes = build_routes(live_config(), reqwest::Client::new());

    let body = serde_json::json!({
        "model": "BAAI/bge-reranker-v2-m3",
        "query": QUERY,
        "documents": [CHEESE_DOC, RELEVANT_DOC, PARIS_DOC],
        "top_n": 2,
        "return_documents": true
    });
    let (status, json) = post_json(&routes, "/jina/rerank", &body).await;

    assert_eq!(status, 200, "body: {json}");
    assert_eq!(json["object"], "rerank");
    assert_eq!(json["model"], "BAAI/bge-reranker-v2-m3");
    let results = json["results"].as_array().unwrap();
    assert_eq!(results.len(), 2, "top_n honored on the Jina dialect");

    // Best match is the relevant doc at input index 1, text echoed back
    assert_eq!(results[0]["index"], 1);
    assert_eq!(results[0]["document"]["text"], RELEVANT_DOC);
    assert!(
        results[0]["relevance_score"].as_f64().unwrap()
            >= results[1]["relevance_score"].as_f64().unwrap(),
        "results must be sorted by relevance descending"
    );
}

#[tokio::test]
#[ignore = "requires live TEI at :4001/:4002"]
async fn live_cohere_rerank_shape() {
    require_live().await;
    let routes = build_routes(live_config(), reqwest::Client::new());

    let body = serde_json::json!({
        "model": "rerank-v3.5",
        "query": QUERY,
        "documents": [RELEVANT_DOC, CHEESE_DOC],
        "top_n": 1
    });
    let (status, json) = post_json(&routes, "/v2/rerank", &body).await;

    assert_eq!(status, 200, "body: {json}");
    assert!(!json["id"].as_str().unwrap().is_empty());
    assert_eq!(json["meta"]["api_version"]["version"], "2");
    let results = json["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["index"], 0);
    assert!(results[0].get("document").is_none());
}

#[tokio::test]
#[ignore = "requires live TEI at :4001/:4002"]
async fn live_rerank_pointed_at_embedding_server_maps_to_502() {
    require_live().await;
    // Deliberately misconfigured: rerank upstream = the embedding instance
    let mut config = live_config();
    config.rerank.endpoint = embed_url();
    let routes = build_routes(config, reqwest::Client::new());

    let body = serde_json::json!({"query": QUERY, "documents": [RELEVANT_DOC]});
    let (status, json) = post_json(&routes, "/jina/rerank", &body).await;

    assert_eq!(status, 502, "body: {json}");
    assert_eq!(json["error"], "upstream_error");
}

// --- Embeddings through the real model ---

#[tokio::test]
#[ignore = "requires live TEI at :4001/:4002"]
async fn live_embeddings_passthrough_openai_shape() {
    require_live().await;
    let routes = build_routes(live_config(), reqwest::Client::new());

    let body = serde_json::json!({
        "input": [QUERY, CHEESE_DOC],
        "model": "anything"
    });
    let (status, json) = post_json(&routes, "/v1/embeddings", &body).await;

    assert_eq!(status, 200, "body: {json}");
    assert_eq!(json["object"], "list");
    let data = json["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);

    let dim0 = data[0]["embedding"].as_array().unwrap().len();
    let dim1 = data[1]["embedding"].as_array().unwrap().len();
    assert!(dim0 > 0, "embedding vector must not be empty");
    assert_eq!(dim0, dim1, "all vectors share the model's dimension");
    assert_eq!(data[0]["index"], 0);
    assert_eq!(data[1]["index"], 1);
    assert!(json["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
#[ignore = "requires live TEI at :4001/:4002"]
async fn live_embeddings_short_alias_matches_v1() {
    require_live().await;
    let routes = build_routes(live_config(), reqwest::Client::new());

    let body = serde_json::json!({"input": [QUERY]});
    let (s1, j1) = post_json(&routes, "/v1/embeddings", &body).await;
    let (s2, j2) = post_json(&routes, "/embeddings", &body).await;

    assert_eq!((s1, s2), (200, 200));
    // Not byte-equality: under concurrent load TEI batches each request
    // differently and float32 reduction order shifts the low bits. Same
    // upstream ⇒ same dimension and (near-)identical direction.
    let v1 = embedding_vec(&j1["data"][0]);
    let v2 = embedding_vec(&j2["data"][0]);
    assert_eq!(v1.len(), v2.len(), "both spellings hit the same model");
    assert!(
        cosine(&v1, &v2) > 0.999,
        "same input on both spellings should embed identically (cosine = {})",
        cosine(&v1, &v2)
    );
}

#[tokio::test]
#[ignore = "requires live TEI at :4001/:4002"]
async fn live_embeddings_are_semantically_meaningful() {
    require_live().await;
    let routes = build_routes(live_config(), reqwest::Client::new());

    // Two paraphrases and one unrelated sentence
    let body = serde_json::json!({
        "input": [
            "Deep learning uses neural networks with many layers.",
            "Neural networks with multiple layers are the basis of deep learning.",
            CHEESE_DOC
        ]
    });
    let (status, json) = post_json(&routes, "/v1/embeddings", &body).await;
    assert_eq!(status, 200);

    let vecs: Vec<Vec<f64>> = json["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(embedding_vec)
        .collect();

    let paraphrase_sim = cosine(&vecs[0], &vecs[1]);
    let unrelated_sim = cosine(&vecs[0], &vecs[2]);
    assert!(
        paraphrase_sim > unrelated_sim,
        "paraphrases ({paraphrase_sim:.3}) should be closer than unrelated text ({unrelated_sim:.3}) — \
         if this fails, the passthrough may be corrupting vectors"
    );
}

// --- Full binary end-to-end ---

/// Kills the spawned proxy even if the test panics.
struct ChildGuard(std::process::Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test]
#[ignore = "requires live TEI at :4001/:4002"]
async fn live_proxy_binary_end_to_end() {
    require_live().await;

    const PORT: u16 = 14987;
    let child = std::process::Command::new(env!("CARGO_BIN_EXE_tei-proxy"))
        .env("RERANKING_HOST", rerank_url())
        .env("EMBEDDING_HOST", embed_url())
        .env("TEI_PROXY_PORT", PORT.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to spawn proxy binary");
    let _guard = ChildGuard(child);

    let base = format!("http://127.0.0.1:{PORT}");
    let client = reqwest::Client::new();

    // Wait for the server to come up
    let mut healthy = false;
    for _ in 0..50 {
        if let Ok(r) = client
            .get(format!("{base}/health"))
            .timeout(Duration::from_millis(500))
            .send()
            .await
        {
            assert_eq!(r.status(), 200, "proxy up but upstreams unhealthy");
            healthy = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(healthy, "proxy binary did not become healthy within 5s");

    // Real OpenWebUI rerank payload over real HTTP
    let rerank: serde_json::Value = client
        .post(format!("{base}/openwebui/rerank"))
        .json(&serde_json::json!({
            "model": "reranker",
            "query": QUERY,
            "documents": [RELEVANT_DOC, CHEESE_DOC],
            "top_n": 2
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let results = rerank["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert!(
        results[0]["relevance_score"].as_f64().unwrap()
            > results[1]["relevance_score"].as_f64().unwrap(),
        "deep-learning doc should outscore cheese: {rerank}"
    );

    // Real embedding over real HTTP
    let embed: serde_json::Value = client
        .post(format!("{base}/v1/embeddings"))
        .json(&serde_json::json!({"input": [QUERY]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(!embed["data"][0]["embedding"].as_array().unwrap().is_empty());
}
