# -------- Stage 1: Builder --------
FROM rust:1.82 AS builder
WORKDIR /app

# Install musl tools (for musl target builds)
RUN apt-get update && apt-get install -y musl-tools musl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Add musl target for static linking
RUN rustup target add x86_64-unknown-linux-musl

# Copy manifests first
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build release binary with musl
RUN cargo build --release --target x86_64-unknown-linux-musl

# -------- Stage 2: Minimal Runtime --------
FROM scratch

# OCI image labels
LABEL org.opencontainers.image.source="https://github.com/delfianto/tei-reranker-proxy"
LABEL org.opencontainers.image.description="Tiny Rust proxy for TEI reranker API"
LABEL org.opencontainers.image.licenses="MIT"

# Copy only the binary, nothing else
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/rerank-proxy /rerank-proxy

# Run as non-root user for security
USER 1000

ENTRYPOINT ["/rerank-proxy"]
