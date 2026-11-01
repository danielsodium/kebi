use crate::tools::ToolRunner;

use super::llm_models::{
    LlmClient, Message, Request, Response, ToolCall, system_message, tool_message, user_message,
};

use std::io::{IsTerminal, Write, stdout};

pub struct Chat {
    model: String,
    tools: ToolRunner,
    history: Vec<Message>,
    client: LlmClient,
    max_tokens: Option<u32>,
}

impl Chat {
    pub fn new(url: String, model: String) -> Self {
        let tools = ToolRunner::new();

        Chat {
            model,
            tools,
            history: vec![system_message(
                "Answer directly and concisely without internal monologue, planning steps, or a chain-of-thought process."
                    .to_string(),
            )],
            client: LlmClient::new(url),
            max_tokens: std::env::var("KEBI_MAX_TOKENS")
                .ok()
                .and_then(|value| value.parse().ok()),
        }
    }

    fn request(&self) -> Request {
        Request {
            model: self.model.clone(),
            messages: self.history.clone(),
            tools: self.tools.tools.clone(),
            max_tokens: self.max_tokens,
        }
    }

    pub fn push_tool_result(&mut self, tool_call_id: &str, content: String) {
        self.history
            .push(tool_message(tool_call_id.to_string(), content));
    }

    pub async fn ask(&mut self, prompt: String) {
        self.history.push(user_message(prompt));
        self.run().await;
    }

    pub async fn run(&mut self) {
        let color = std::io::stdout().is_terminal();
        let mut finished = false;

        for _ in 0..25 {
            let mut reasoning_deltas: Vec<String> = Vec::new();
            let mut answer_deltas: Vec<String> = Vec::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();

            let outcome = match self
                .client
                .stream(self.request(), |event| match event {
                    Response::ReasoningDelta { delta } => {
                        reasoning_deltas.push(delta);
                        print_delta(color, &reasoning_deltas)
                    }
                    Response::AnswerDelta { delta } => {
                        answer_deltas.push(delta);
                        print_delta(false, &answer_deltas);
                    }
                    Response::ToolCall { tool } => tool_calls.push(tool),
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
            }

            match outcome.finish_reason.as_deref() {
                Some("length") => eprintln!(
                    "\nResponse truncated: the model hit the token limit before finishing. Set KEBI_MAX_TOKENS to a larger value."
                ),
                Some(_) => {}
                None if outcome.error.is_none() => {
                    eprintln!("\nStream ended before the model finished (connection closed early).")
                }
                None => {}
            }

            if !tool_calls.is_empty() {
                for call in &tool_calls {
                    println!("Ran {}...", call.name);
                    let mut result = self
                        .tools
                        .run(&call.name, &call.args)
                        .unwrap_or_else(|err| err.to_string());

                    match self.tools.refresh() {
                        Ok(outcome) => {
                            if !outcome.errors.is_empty() {
                                result.push_str("\n\nTool reload errors:");
                                for err in outcome.errors {
                                    result.push_str(&format!("\n- {err}"));
                                }
                            }
                        }
                        Err(err) => {
                            result.push_str(&format!("\n\nFailed to refresh tools: {err}"));
                        }
                    }

                    println!("Tool returned: {}", result);
                    self.push_tool_result(&call.id, result);
                }
            } else {
                println!();
                finished = true;
                break;
            }
        }

        if !finished {
            eprintln!("Stopped after 25 tool iterations without a final answer.");
        }
    }
}

fn print_delta(color: bool, deltas: &[String]) {
    let delta = &deltas[deltas.len() - 1];
    if color {
        print!("\x1b[2m{delta}\x1b[0m");
    } else {
        print!("{delta}");
    }
    stdout().flush().unwrap()
}
