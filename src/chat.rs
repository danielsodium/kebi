use crate::embedding::Embedder;
use crate::llm_models::{LlmClient, Message, Request, Response, system_message, user_message};
use crate::memory::{Memory, MemoryOp};
use crate::tools::{ToolCall, ToolManager};

use serde::Deserialize;
use std::io::{Write, stdout};
use std::sync::Arc;

static PROMPT: &str = r#"
### Decision Order (Follow Every Turn)

1. `memory_ops` — Save Only High-Value Facts in Telegraphic Format
   - WHAT TO SAVE: Only lasting personal facts, preferences, constraints, relationships, ongoing projects, or key dates from the user's latest message that are NOT already in `<retrieved_context>` or history.
   - WHAT TO IGNORE: Never save greetings, chitchat, one-off questions, temporary states ("I'm tired", "headed out"), general world facts, or assistant replies. When running after `Tool ... returned:` or when no lasting fact was shared, set `"memory_ops": []`.
   - TOKEN-EFFICIENT FORMAT: In `{"op": "mem_save", "content": "..."}`, write `content` as ultra-compact `<topic>: <keywords/value>` shorthand (3–7 words max):
     - Strip "User", "The user is", articles (`a`, `an`, `the`), verbs like `is/has/likes`, and punctuation filler.
     - Good: `"allergy: peanuts"`, `"dog: Mochi"`, `"location: Seattle"`, `"coffee: dark roast, oat milk"`, `"stack: Rust, SQLite"`.
     - Bad: `"The user is allergic to peanuts and cannot eat them."` (too wordy).
   - Save immediately on the turn the user mentions the fact, even if also calling a tool.

2. `tool_call` — Check Context First, Search Memory When Needed
   - First check `<retrieved_context>` and the conversation history.
   - If `<retrieved_context>` or history already contains the needed information, set `"tool_call": null`.
   - If answering requires user background, preferences, or past facts NOT in `<retrieved_context>`, call `search-memory` BEFORE guessing or asking the user:
     `"tool_call": {"name": "search-memory", "args": {"query": "<short keyword query>"}}, "response": null`
   - NO REPEAT CALLS: Never call `search-memory` with the same or redundant query twice. Once `search-memory` returns `[SEARCH]`, use those results to reply (or ask the user if nothing matched).

3. `response` — Concise Final Reply
   - `response` and `tool_call` are strictly mutually exclusive:
     - Calling a tool: `"tool_call": {"name": "...", "args": {...}}`, `"response": null`
     - Replying to user: `"tool_call": null`, `"response": "<your concise reply>"`
   - Keep answers brief and direct.
   - When asking the user questions, ask only ONE question at a time (never a list of questions).

### Output Examples

User shares a new fact and asks something not in `<retrieved_context>`:
{"memory_ops":[{"op":"mem_save","content":"allergy: peanuts"}],"response":null,"tool_call":{"name":"search-memory","args":{"query":"favorite restaurants dietary"}}}

User shares a new fact and no tool is needed:
{"memory_ops":[{"op":"mem_save","content":"location: Seattle"},{"op":"mem_save","content":"coffee: dark roast, oat milk"}],"response":"Got it! What neighborhood in Seattle are you looking for coffee spots in?","tool_call":null}

Replying after a tool result or when no lasting user fact was shared:
{"memory_ops":[],"response":"Your dentist appointment is Thursday at 2 PM.","tool_call":null}
"#;

const MAX_TOOL_ROUNDS: usize = 500;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Answer {
    #[serde(default)]
    memory_ops: Vec<MemoryOp>,
    #[serde(default)]
    response: Option<String>,
    #[serde(default)]
    tool_call: Option<ToolCall>,
}

pub struct Chat {
    model: String,
    history: Vec<Message>,
    client: LlmClient,
    max_tokens: Option<u32>,
    tools: ToolManager,
    memory: Memory,
}

impl Chat {
    pub fn new(url: String, model: String, embedder: &Arc<Embedder>) -> Self {
        let tools = ToolManager::new();
        let memory = Memory::new(
            embedder,
            "/home/hna/.local/share/kebi/memory.db".to_string(),
        )
        .unwrap();

        Chat {
            model,
            history: vec![],
            client: LlmClient::new(url),
            max_tokens: std::env::var("KEBI_MAX_TOKENS")
                .ok()
                .and_then(|value| value.parse().ok()),
            tools,
            memory,
        }
    }

    fn request(&self) -> Request {
        Request {
            model: self.model.clone(),
            messages: self.history.clone(),
            max_tokens: self.max_tokens,
        }
    }

    pub async fn ask(&mut self, prompt: String) {
        if self.history.is_empty() {
            let context = self.memory.get_context(&prompt).unwrap();
            let context_block = if context.is_empty() {
                "<retrieved_context>\n(none)\n</retrieved_context>".to_string()
            } else {
                format!(
                    "<retrieved_context>\n- {}\n</retrieved_context>",
                    context.join("\n- ")
                )
            };
            let system_prompt = format!(
                "You are a sharp, resourceful personal assistant. Output ONLY valid JSON matching the schema.\n\n{}{}\n{}",
                self.tools.prompt(),
                context_block,
                PROMPT
            );
            self.history.push(system_message(system_prompt));
        }
        self.history.push(user_message(prompt));
        self.run().await;
    }

    pub async fn run(&mut self) {
        //let color = std::io::stdout().is_terminal();

        for _ in 0..MAX_TOOL_ROUNDS {
            let mut reasoning_deltas: Vec<String> = Vec::new();
            let mut answer_deltas: Vec<String> = Vec::new();

            let outcome = match self
                .client
                .stream::<Answer, _>(self.request(), self.tools.response_schema(), |event| {
                    match event {
                        Response::ReasoningDelta { delta } => {
                            reasoning_deltas.push(delta);
                        }
                        Response::ContentDelta { delta } => {
                            answer_deltas.push(delta);
                        }
                    }
                })
                .await
            {
                Ok(outcome) => outcome,
                Err(err) => {
                    eprintln!("\nRequest failed: {err}");
                    return;
                }
            };

            self.history.push(outcome.message);

            if let Some(err) = &outcome.error {
                eprintln!("\nStream interrupted: {err}");
                return;
            }

            match outcome.finish_reason.as_deref() {
                Some("length") => {
                    eprintln!(
                        "\nResponse truncated: the model hit the token limit before finishing. Set KEBI_MAX_TOKENS to a larger value."
                    );
                    return;
                }
                Some(_) => {}
                None => {
                    eprintln!(
                        "\nStream ended before the model finished (connection closed early)."
                    );
                    return;
                }
            }

            if let Some(err) = &outcome.parse_error {
                eprintln!("\nFailed to parse structured output: {err}");
                return;
            }

            let Some(Answer {
                memory_ops,
                response,
                tool_call,
            }) = outcome.parsed
            else {
                eprintln!("\nThe model did not return a structured answer.");
                return;
            };

            if !reasoning_deltas.is_empty() {
                println!();
            }

            if !memory_ops.is_empty() {
                for mem in memory_ops {
                    self.memory
                        .insert(&mem.content)
                        .unwrap_or_else(|e| eprintln!("Failed to save to memory: {}", e));
                }
            }

            if let Some(tool_call) = tool_call {
                let result = self.tools.call(&tool_call, &mut self.memory).await;
                self.history.push(user_message(format!(
                    "Tool `{}` returned:\n{}\n\nSet `memory_ops` to `[]`. Do not repeat the same tool call. Set `tool_call` to `null` and write your `response` unless a different tool query is needed.",
                    tool_call.name, result
                )));
            }

            if let Some(response) = response {
                println!("{response}");
                return;
            }

            eprintln!("\nThe model returned neither an answer nor a tool call.");
            return;
        }

        eprintln!("Stopped after {MAX_TOOL_ROUNDS} tool rounds without a final answer.");
    }
}

#[allow(dead_code)]
fn print_delta(color: bool, deltas: &[String]) {
    let delta = &deltas[deltas.len() - 1];
    if color {
        print!("\x1b[2m{delta}\x1b[0m");
    } else {
        print!("{delta}");
    }
    stdout().flush().unwrap()
}
