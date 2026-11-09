pub mod agents;
pub mod backend;
pub mod chat;
pub mod embedding;
pub mod llm_models;
pub mod memory;
pub mod tools;

use crate::agents::AgentManager;
use crate::backend::start_router;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (be_tx, be_rx) = mpsc::channel(32);
    tokio::spawn(AgentManager::run(be_rx));
    start_router(be_tx).await;
}
