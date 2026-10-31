pub mod llm_models;
pub mod models;
use models::Chat;

#[tokio::main]
async fn main() {
    let mut chat = Chat::new(
        "http://localhost:8080/v1".to_string(),
        std::env::var("KEBI_MODEL").unwrap_or_else(|_| "default".to_string()),
    );
    chat.ask("Can you look up the latest version of the openai docs?".to_string())
        .await;
}
