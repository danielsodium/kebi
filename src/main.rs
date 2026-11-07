pub mod agents;
pub mod chat;
pub mod llm_models;
pub mod memory;
pub mod tools;
use agents::Agent;
use std::io::{self, Write};

#[tokio::main]
async fn main() {
    let mut agent = Agent::new("kebi".to_string());
    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        let mut prompt = String::new();
        io::stdin()
            .read_line(&mut prompt)
            .expect("Failed to get prompt");

        agent.chat(prompt).await;
    }
}
