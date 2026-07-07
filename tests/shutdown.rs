//! SIGTERM must terminate the server promptly and cleanly. Docker sends
//! SIGTERM on stop, and as PID 1 in the scratch image the process only dies
//! if it installs a handler — an unhandled SIGTERM is ignored outright and
//! the container rides out the full stop grace period until SIGKILL.
//!
//! Unlike the live_tei suite this needs no running TEI: the proxy binds its
//! port regardless of upstream reachability.

use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::time::{Duration, Instant};

#[test]
fn sigterm_shuts_down_promptly_and_cleanly() {
    // Bind-then-drop to pick a port that is almost certainly free
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();

    let mut child = Command::new(env!("CARGO_BIN_EXE_tei-proxy"))
        .env("TEI_PROXY_PORT", port.to_string())
        .spawn()
        .expect("failed to spawn proxy binary");

    // Wait for the server to accept connections
    let start = Instant::now();
    while TcpStream::connect(("127.0.0.1", port)).is_err() {
        if start.elapsed() > Duration::from_secs(10) {
            let _ = child.kill();
            panic!("proxy never started listening on port {}", port);
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    let kill_status = Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .expect("failed to run kill");
    assert!(kill_status.success(), "kill -TERM failed");

    // Graceful shutdown must be prompt and exit 0; death by signal (the
    // pre-handler behavior outside a container) reports unsuccessful status
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(
                status.success(),
                "expected clean exit after SIGTERM, got: {}",
                status
            );
            return;
        }
        if start.elapsed() > Duration::from_secs(5) {
            let _ = child.kill();
            panic!("proxy did not exit within 5s of SIGTERM");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}
