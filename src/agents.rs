use crate::chat;
use chat::Chat;

pub struct Agent {
    name: String,
    chats: Vec<Chat>,
}

impl Agent {
    pub fn new(name: String) -> Self {
        Agent {
            name,
            chats: Vec::new(),
        }
    }

    pub async fn chat(&mut self, prompt: String) {
        if self.chats.is_empty() {
            let chat = Chat::new(
                "http://localhost:8080/v1".to_string(),
                std::env::var("KEBI_MODEL").unwrap_or_else(|_| "default".to_string()),
            );
            self.chats.push(chat);
        }
        self.chats[0].ask(prompt.trim().to_string()).await;
    }
}
