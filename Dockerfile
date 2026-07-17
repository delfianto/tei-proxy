# -------- Stage 1: Builder --------
FROM rust:1.97 AS builder
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

# Copy only the binary, nothing else
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/tei-proxy /tei-proxy

# Run as non-root user for security
USER 1000

# Exec-form only: scratch has no shell. The binary re-invokes itself in
# self-check mode and probes its own /health (deep check incl. upstreams).
# Long start-period tolerates TEI upstreams loading models at stack boot.
HEALTHCHECK --interval=10s --timeout=10s --retries=5 --start-period=120s \
    CMD ["/tei-proxy", "--healthcheck"]

ENTRYPOINT ["/tei-proxy"]
