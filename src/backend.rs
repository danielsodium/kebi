use crate::agents::BackendMessage;
use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use serde::Deserialize;
use tokio::sync::mpsc;

#[derive(Clone)]
struct AppState {
    tx: mpsc::Sender<BackendMessage>,
}

pub async fn start_router(tx: mpsc::Sender<BackendMessage>) {
    let state = AppState { tx };

    let app = Router::new()
        .route("/api/messages", post(send_message))
        .route("/", get(root))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!(
        "Server listening on http://{}",
        listener.local_addr().unwrap()
    );

    axum::serve(listener, app).await.unwrap();
}

#[derive(Deserialize)]
struct MessagePayload {
    message: String,
}

async fn send_message(
    State(state): State<AppState>,
    Json(payload): Json<MessagePayload>,
) -> String {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let _ = state
        .tx
        .send(BackendMessage::Message {
            message: payload.message,
            respond_to: tx,
        })
        .await;

    rx.await.unwrap_or_default()
}

async fn root() -> &'static str {
    "Hello from the Rust backend!"
}
