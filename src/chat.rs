use crate::llm_models::{LlmClient, Message, Request, Response, system_message, user_message};
use crate::memory::{Memory, MemoryOp};
use crate::tools::{ToolCall, ToolManager};

use serde::Deserialize;
use std::io::{Write, stdout};

static PROMPT: &str = r#"

### Guidelines
- When asking questions to the user, ask them one at a time. Do not splurt out a list of questions.
- Keep answers as concise as possible, not including unnecessary information.
- Constantly save any information about the user you get.
### Memory & Onboarding Rules
MEMORY_OPS:
 - Save any new information that the user gives. Do not save user information already in memory.
 - To save a specific detail: use `op`="mem_save"  and write a concise 1-sentence fact in `content`.
 - Always use the search-memory tool to look for relevant information before asking the user.

### Execution & Anti-Loop Rules
1. CHECK FIRST: Read <core_memory> and <retrieved_context> before calling `mem_search`.
2. NO REPEAT CALLS: Never call the same tool with identical args twice.
3. ENVELOPE:
   - Tool call: `action`="tool", set `tool_name` and `args`, `reply`=null.
   - Answer user: `action`="reply", set `reply`, `tool_name`=null, `args`=null.
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
    pub fn new(url: String, model: String) -> Self {
        let tools = ToolManager::new();
        let memory = Memory::new("/home/hna/.local/share/kebi/memory.db".to_string()).unwrap();

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
            let context_block = format!(
                "<retrieved_context>\n{}\n</retrieved_context>",
                context.join("\n")
            );
            let system_prompt = format!(
                "You are a sharp, resourceful personal assistant. Output ONLY valid JSON matching the schema.\n{}<core_memory></core_memory>\n\n{}{}",
                self.tools.prompt(),
                context_block,
                PROMPT
            );
            println!("[SYSTEM] {}", system_prompt);
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

            println!("[RAW]: {}", answer_deltas.concat());

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
                    println!("[MEM SAVE] {}", &mem.content);
                }
            }

            if let Some(tool_call) = tool_call {
                if response.is_some() {
                    eprintln!(
                        "\nThe model returned both an answer and a tool call; using the tool call."
                    );
                }

                println!("Calling {}...", tool_call.name);
                let result = self.tools.call(&tool_call, &mut self.memory).await;
                println!("Tool returned: {result}");
                self.history.push(user_message(format!(
                    "Tool `{}` returned:\n{}\n\nContinue. Give the final answer if you have enough information.",
                    tool_call.name, result
                )));
                continue;
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
