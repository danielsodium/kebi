use std::sync::Arc;
use tokio::sync::oneshot::Sender;

use crate::llm_models::LlmClient;
use crate::memory::Memory;
use crate::tools::ToolManager;
use crate::{chat, embedding::Embedder};
use chat::Chat;
use tokio::sync::mpsc;
use tokio::task::JoinSet;

pub struct AgentManager {
    handles: JoinSet<()>,
    ctx: Arc<AgentContext>,
    be_rx: mpsc::Receiver<BackendMessage>,
}

pub enum AgentMessage {
    Response { message: String },
}

pub enum BackendMessage {
    Message {
        message: String,
        respond_to: tokio::sync::oneshot::Sender<String>,
    },
    End,
}

pub struct AgentContext {
    pub model: String,
    pub client: LlmClient,
    pub max_tokens: Option<u32>,
    pub tools: ToolManager,
    pub memory: Memory,
}

impl AgentManager {
    pub fn new(be_rx: mpsc::Receiver<BackendMessage>) -> Self {
        let tools = ToolManager::new();
        let embedder = Arc::new(Embedder::new().unwrap());
        let memory = Memory::new(
            &embedder,
            "/home/hna/.local/share/kebi/memory.db".to_string(),
        )
        .unwrap();
        let ctx = AgentContext {
            model: std::env::var("KEBI_MODEL").unwrap_or_else(|_| "default".to_string()),
            client: LlmClient::new("http://localhost:8080/v1".to_string()),
            max_tokens: std::env::var("KEBI_MAX_TOKENS")
                .ok()
                .and_then(|value| value.parse().ok()),
            tools,
            memory,
        };
        let agent_manager = AgentManager {
            handles: JoinSet::new(),
            ctx: Arc::new(ctx),
            be_rx,
        };
        agent_manager
    }

    pub async fn run(be_rx: mpsc::Receiver<BackendMessage>) {
        let mut am = AgentManager::new(be_rx);

        loop {
            tokio::select! {
                Some(res) = am.handles.join_next(), if !am.handles.is_empty() => {
                    if let Err(e) = res {
                        eprintln!("[ERROR] {e}");
                    }
                }
                Some(msg) = am.be_rx.recv() => {
                    match msg {
                        BackendMessage::Message{message, respond_to} => {
                            am.handles.spawn(chat(message, am.ctx.clone(), respond_to));
                        }
                        BackendMessage::End => break

                    }
                }
                else => break
            }
        }
        while let Some(res) = am.handles.join_next().await {
            res.unwrap();
        }
    }
}

pub async fn chat(prompt: String, ctx: Arc<AgentContext>, respond_to: Sender<String>) {
    let mut chat = Chat::new(ctx);
    let res = chat
        .ask(prompt.trim().to_string())
        .await
        .unwrap_or_else(|e| format!("An unexpected error occured: {e}"));
    let _ = respond_to.send(res);
}
