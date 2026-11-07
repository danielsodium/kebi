use std::io::{self, Write};
use std::{collections::HashMap, sync::Arc};

use crate::{chat, embedding::Embedder};
use chat::Chat;
use tokio::sync::mpsc;
use tokio::task::JoinSet;

pub struct AgentManager {
    interactive: String,
    embedder: Arc<Embedder>,
    handles: JoinSet<()>,
    tx: mpsc::Sender<AgentMessage>,
    rx: mpsc::Receiver<AgentMessage>,
}

pub struct AgentMessage {}

#[allow(dead_code)]
pub struct Agent {
    name: String,
    chats: Vec<Chat>,
    embedder: Arc<Embedder>,
    tx: mpsc::Sender<AgentMessage>,
}

impl AgentManager {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel(32);
        let interactive_name = "kebi".to_string();
        let agent_manager = AgentManager {
            embedder: Arc::new(Embedder::new().unwrap()),
            handles: JoinSet::new(),
            interactive: interactive_name.clone(),
            tx,
            rx,
        };
        agent_manager
    }

    pub async fn run(&mut self) {
        let interactive_agent =
            Agent::new(self.interactive.clone(), &self.embedder, self.tx.clone());
        self.handles.spawn(run_interactive(interactive_agent));
        while let Some(res) = self.handles.join_next().await {
            res.unwrap();
        }
    }
}

impl Agent {
    pub fn new(name: String, embedder: &Arc<Embedder>, tx: mpsc::Sender<AgentMessage>) -> Self {
        Agent {
            name,
            chats: Vec::new(),
            embedder: embedder.clone(),
            tx,
        }
    }

    pub async fn chat(&mut self, prompt: String) {
        if self.chats.is_empty() {
            let chat = Chat::new(
                "http://localhost:8080/v1".to_string(),
                std::env::var("KEBI_MODEL").unwrap_or_else(|_| "default".to_string()),
                &self.embedder,
            );
            self.chats.push(chat);
        }
        self.chats[0].ask(prompt.trim().to_string()).await;
    }
}

pub async fn run_interactive(mut agent: Agent) {
    print!("> ");
    io::stdout().flush().unwrap();
    let mut prompt = String::new();
    io::stdin()
        .read_line(&mut prompt)
        .expect("Failed to get prompt");

    agent.chat(prompt).await;
}
