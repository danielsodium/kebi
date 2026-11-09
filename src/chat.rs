use crate::llm_models::{Message, Request, Response, system_message, user_message};
use crate::memory::MemoryOp;
use crate::tools::ToolCall;

use crate::agents::AgentContext;
use anyhow::{Result, anyhow};
use serde::Deserialize;
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
    history: Vec<Message>,
    ctx: Arc<AgentContext>,
}

impl Chat {
    pub fn new(ctx: Arc<AgentContext>) -> Self {
        Chat {
            history: Vec::new(),
            ctx,
        }
    }

    fn request(&self) -> Request {
        Request {
            model: self.ctx.model.clone(),
            messages: self.history.clone(),
            max_tokens: self.ctx.max_tokens,
        }
    }

    pub async fn ask(&mut self, prompt: String) -> Result<String> {
        if self.history.is_empty() {
            let context = self.ctx.memory.get_context(&prompt).await.unwrap();
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
                self.ctx.tools.prompt(),
                context_block,
                PROMPT
            );
            self.history.push(system_message(system_prompt));
        }
        self.history.push(user_message(prompt));
        self.run().await
    }

    pub async fn run(&mut self) -> Result<String> {
        for _ in 0..MAX_TOOL_ROUNDS {
            let mut reasoning_deltas: Vec<String> = Vec::new();
            let mut answer_deltas: Vec<String> = Vec::new();

            let outcome = match self
                .ctx
                .client
                .stream::<Answer, _>(self.request(), self.ctx.tools.response_schema(), |event| {
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
                    return Err(anyhow!("Request failed: {err}"));
                }
            };

            self.history.push(outcome.message);

            if let Some(err) = &outcome.error {
                return Err(anyhow!("Stream interrupted: {err}"));
            }

            match outcome.finish_reason.as_deref() {
                Some("length") => {
                    return Err(anyhow!(
                        "\nResponse truncated: the model hit the token limit before finishing. Set KEBI_MAX_TOKENS to a larger value."
                    ));
                }
                Some(_) => {}
                None => {
                    return Err(anyhow!(
                        "\nStream ended before the model finished (connection closed early)."
                    ));
                }
            }

            if let Some(err) = &outcome.parse_error {
                return Err(anyhow!("\nFailed to parse structured output: {err}"));
            }

            let Some(Answer {
                memory_ops,
                response,
                tool_call,
            }) = outcome.parsed
            else {
                return Err(anyhow!("\nThe model did not return a structured answer."));
            };

            if !reasoning_deltas.is_empty() {
                println!();
            }

            if !memory_ops.is_empty() {
                for mem in memory_ops {
                    self.ctx
                        .memory
                        .insert(&mem.content)
                        .await
                        .unwrap_or_else(|e| eprintln!("Failed to save to memory: {}", e));
                }
            }

            if let Some(tool_call) = tool_call {
                let result = self.ctx.tools.call(&tool_call, &self.ctx.memory).await;
                self.history.push(user_message(format!(
                    "Tool `{}` returned:\n{}\n\nSet `memory_ops` to `[]`. Do not repeat the same tool call. Set `tool_call` to `null` and write your `response` unless a different tool query is needed.",
                    tool_call.name, result
                )));
            }

            if let Some(response) = response {
                return Ok(response);
            }

            return Err(anyhow!(
                "\nThe model returned neither an answer nor a tool call."
            ));
        }

        Err(anyhow!(
            "Stopped after {MAX_TOOL_ROUNDS} tool rounds without a final answer."
        ))
    }
}
