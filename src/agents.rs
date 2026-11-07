use std::io::{self, Write};
use std::{collections::HashMap, sync::Arc};

use crate::{chat, embedding::Embedder};
use chat::Chat;

pub struct AgentManager {
    agents: HashMap<String, Agent>,
    embedder: Arc<Embedder>,
    interactive: String,
}

#[allow(dead_code)]
pub struct Agent {
    name: String,
    chats: Vec<Chat>,
    embedder: Arc<Embedder>,
}

impl AgentManager {
    pub fn new() -> Self {
        let interactive_name = "kebi".to_string();
        let mut agent_manager = AgentManager {
            agents: HashMap::new(),
            embedder: Arc::new(Embedder::new().unwrap()),
            interactive: interactive_name.clone(),
        };
        agent_manager.create_agent(interactive_name);
        agent_manager
    }

    pub fn create_agent(&mut self, name: String) {
        let agent = Agent::new(name, &self.embedder);
        self.agents.insert(agent.name.clone(), agent);
    }

    pub async fn run(&mut self) {
        loop {
            print!("> ");
            io::stdout().flush().unwrap();
            let mut prompt = String::new();
            io::stdin()
                .read_line(&mut prompt)
                .expect("Failed to get prompt");

            self.agents
                .get_mut(&self.interactive)
                .unwrap()
                .chat(prompt)
                .await;
        }
    }
}

impl Agent {
    pub fn new(name: String, embedder: &Arc<Embedder>) -> Self {
        Agent {
            name,
            chats: Vec::new(),
            embedder: embedder.clone(),
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
