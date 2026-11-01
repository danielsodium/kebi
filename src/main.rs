pub mod llm_models;
pub mod models;
pub mod tools;
use models::Chat;
use std::io::{self, Write};

#[tokio::main]
async fn main() {
    let mut chat = Chat::new(
        "http://localhost:8080/v1".to_string(),
        std::env::var("KEBI_MODEL").unwrap_or_else(|_| "default".to_string()),
    );
    print!("> ");
    io::stdout().flush().unwrap();
    let mut prompt = String::new();
    io::stdin()
        .read_line(&mut prompt)
        .expect("Failed to get prompt");

    chat.ask(prompt.trim().to_string()).await;
}
