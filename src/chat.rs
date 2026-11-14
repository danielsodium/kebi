use crate::llm_models::{Message, Request, Response, system_message, user_message};
use crate::memory::MemoryOp;
use crate::tools::ToolCall;

use crate::agents::AgentContext;
use anyhow::{Result, anyhow};
use serde::Deserialize;
use std::sync::Arc;

static PROMPT: &str = r#"
### Decision Order (Follow Every Turn)

1. `memory_ops` — Save, Update, or Delete High-Value Facts
   - WHAT TO SAVE (`mem_save`): Only lasting personal facts, preferences, constraints, relationships, ongoing projects, or key dates from the user's latest message that are NOT already in `<retrieved_context>` or history.
   - WHAT TO UPDATE OR DELETE (`mem_update` / `mem_delete`): If the user corrects or changes an existing memory shown with `[id=N]` in `<retrieved_context>` or tool results, use `{"op": "mem_update", "id": N, "content": "..."}` to overwrite it, or `{"op": "mem_delete", "id": N}` if it is no longer true.
   - WHAT TO IGNORE: Never save greetings, chitchat, one-off questions, temporary states ("I'm tired", "headed out"), general world facts, or assistant replies. When running after `<tool_result>` or when no lasting fact was shared, set `"memory_ops": []`.
   - ATOMIC SENTENCE FORMAT: In `mem_save` and `mem_update`, write `content` as a clear, self-contained atomic sentence with parenthetical domain/category keywords at the end to aid semantic and keyword search:
     - Store one distinct fact per entry so each memory can stand on its own.
     - Good: `"User has a severe allergy to peanuts (dietary restriction, food)."`, `"User's dog is named Mochi (pet, dog)."`, `"User lives in Seattle (location, home)."`, `"User prefers dark roast coffee with oat milk (coffee, beverage preference)."`, `"User's primary tech stack is Rust and SQLite (programming, software stack)."`
     - Bad: `"allergy: peanuts"` (too fragmented for semantic search) or multi-topic paragraphs combining unrelated facts.
   - Apply memory operations immediately on the turn the user mentions the fact, even if also calling a tool.

2. `tool_call` — Check Context First, Search or List Memories When Needed
   - First check `<retrieved_context>` and the conversation history.
   - If `<retrieved_context>` or history already contains the needed information, set `"tool_call": null`.
   - If answering requires user background, preferences, or past facts NOT in `<retrieved_context>`, call `search-memory` (for specific topics) or `list-memories` (for a broad overview) BEFORE guessing or asking the user:
     `"tool_call": {"name": "search-memory", "args": {"query": "<focused search query>"}}, "response": null`
   - REFINING SEARCHES: Never call `search-memory` with the exact same query twice. If the first search returns no matching memories or incomplete context, you may try a broader query, alternative synonyms/keywords, or `list-memories` before replying.

3. `response` — Concise Final Reply
   - `response` and `tool_call` are strictly mutually exclusive:
     - Calling a tool: `"tool_call": {"name": "...", "args": {...}}`, `"response": null`
     - Replying to user: `"tool_call": null`, `"response": "<your concise reply>"`
   - Keep answers brief and direct.
   - When asking the user questions, ask only ONE question at a time (never a list of questions).

### Output Examples

User shares a new fact and asks something not in `<retrieved_context>`:
{"memory_ops":[{"op":"mem_save","content":"User has a severe allergy to peanuts (dietary restriction, food)."}],"tool_call":{"name":"search-memory","args":{"query":"favorite restaurants dietary preferences"}},"response":null}

User shares a new fact and no tool is needed:
{"memory_ops":[{"op":"mem_save","content":"User lives in Seattle (location, home)."},{"op":"mem_save","content":"User prefers dark roast coffee with oat milk (coffee, beverage preference)."}],"tool_call":null,"response":"Got it! What neighborhood in Seattle are you looking for coffee spots in?"}

Replying after a tool result or when no lasting user fact was shared:
{"memory_ops":[],"tool_call":null,"response":"Your dentist appointment is Thursday at 2 PM."}
"#;

const MAX_TOOL_ROUNDS: usize = 500;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Answer {
    #[serde(default)]
    memory_ops: Vec<MemoryOp>,
    #[serde(default)]
    tool_call: Option<ToolCall>,
    #[serde(default)]
    response: Option<String>,
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
                let formatted: Vec<String> = context.iter().map(|m| m.format()).collect();
                format!(
                    "<retrieved_context>\n- {}\n</retrieved_context>",
                    formatted.join("\n- ")
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

            if !memory_ops.is_empty() {
                for mem in &memory_ops {
                    self.ctx
                        .memory
                        .apply(mem)
                        .await
                        .unwrap_or_else(|e| eprintln!("Failed to apply memory op: {}", e));
                }
            }

            if let Some(tool_call) = tool_call {
                println!("[TOOL CALL] {}", &tool_call.name);
                let result = self.ctx.tools.call(&tool_call, &self.ctx.memory).await;
                println!("[TOOL CALL] RESPONSE : {}", &result);
                self.history.push(user_message(format!(
                    "<tool_result name=\"{}\">\n{}\n</tool_result>\n\
                    Set `memory_ops` to `[]`. If the result above answers the question, set `tool_call` to `null` and write `response`. \
                    If no memories matched or key details are still missing, you may try a broader or synonym search query or `list-memories` before answering.",
                    tool_call.name, result
                )));
                continue;
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
