pub mod agents;
pub mod chat;
pub mod embedding;
pub mod llm_models;
pub mod memory;
pub mod tools;

use crate::agents::AgentManager;

#[tokio::main]
async fn main() {
    let mut agents = AgentManager::new();
    agents.run().await;
}
