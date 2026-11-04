use super::llm_models::{LlmClient, Message, Request, Response, system_message, user_message};
use super::tools::{ToolCall, ToolManager};

use schemars::JsonSchema;
use serde::Deserialize;
use std::io::{IsTerminal, Write, stdout};

#[derive(Debug, Deserialize, JsonSchema)]
struct Answer {
    answer: Option<String>,
    tool_call: Option<ToolCall>,
}

pub struct Chat {
    model: String,
    history: Vec<Message>,
    client: LlmClient,
    max_tokens: Option<u32>,
    tools: ToolManager,
}

impl Chat {
    pub fn new(url: String, model: String) -> Self {
        let tools = ToolManager::new();
        Chat {
            model,
            history: vec![system_message(
                "You are a concise, capable AI assistant. Answer directly without filler.\n\n"
                    .to_string()
                    + &tools.prompt()
                    + "# Output Format\nRespond directly:\n{\"reply\": \"concise answer\"}\nOr call a tool:\n{\"tool\": \"tool_name\", \"args\": {\"arg1\": \"val\"}}",
            )],
            client: LlmClient::new(url),
            max_tokens: std::env::var("KEBI_MAX_TOKENS")
                .ok()
                .and_then(|value| value.parse().ok()),
            tools,
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
        self.history.push(user_message(prompt));
        self.run().await;
    }

    pub async fn run(&mut self) {
        let color = std::io::stdout().is_terminal();
        let mut reasoning_deltas: Vec<String> = Vec::new();

        let outcome = match self
            .client
            .stream::<Answer, _>(self.request(), |event| {
                if let Response::ReasoningDelta { delta } = event {
                    reasoning_deltas.push(delta);
                    print_delta(color, &reasoning_deltas);
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
            Some("length") => eprintln!(
                "\nResponse truncated: the model hit the token limit before finishing. Set KEBI_MAX_TOKENS to a larger value."
            ),
            Some(_) => {}
            None => {
                eprintln!("\nStream ended before the model finished (connection closed early).");
                return;
            }
        }

        if let Some(err) = &outcome.parse_error {
            eprintln!("\nFailed to parse structured output: {err}");
            return;
        }

        let Some(Answer { answer, tool_call }) = outcome.parsed else {
            eprintln!("\nThe model did not return a structured answer.");
            return;
        };

        if !reasoning_deltas.is_empty() {
            println!();
        }
        if let Some(answer) = answer {
            println!("{answer}");
        }
        if let Some(tool_call) = tool_call {
            let tn = tool_call.name;
            println!("{tn}");
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
