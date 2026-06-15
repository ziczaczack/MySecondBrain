# Rust Async Runtime: Tokio Deep Dive

Tokio is the most widely used async runtime for Rust, providing the executor, I/O driver, and timer infrastructure needed to run `async`/`await` code.

## Core concepts

- **Runtime**: `tokio::runtime::Runtime` drives the event loop. Use `#[tokio::main]` for convenience.
- **Tasks**: Spawned with `tokio::spawn`, they are lightweight green threads scheduled by Tokio's work-stealing executor.
- **Futures**: Rust's `async fn` returns a `Future` that does nothing until polled by the runtime.

## Example

```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let a = tokio::spawn(async {
        sleep(Duration::from_millis(100)).await;
        "task a done"
    });

    let b = tokio::spawn(async { "task b done" });

    let (ra, rb) = tokio::join!(a, b).unwrap(); // wait for both
    println!("{ra}, {rb}");
}
```

## Multi-threaded vs current-thread

| Flavor | Threads | Use case |
|---|---|---|
| `Runtime::new()` | # of CPU cores | High-throughput servers |
| `Builder::new_current_thread()` | 1 | Embedded, tests, CLI tools |

## Common pitfalls

- Blocking inside async code stalls the executor thread — use `tokio::task::spawn_blocking` for CPU-heavy or blocking I/O work.
- `tokio::sync::Mutex` should be preferred over `std::sync::Mutex` when the critical section spans `.await` points.
