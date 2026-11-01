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
        }
    }

    fn request(&self) -> Request {
        Request {
            model: self.model.clone(),
            messages: self.history.clone(),
            tools: self.tools.tools.clone(),
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

        for _ in 0..25 {
            let mut reasoning_deltas: Vec<String> = Vec::new();
            let mut answer_deltas: Vec<String> = Vec::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();

            let bot_response = self
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
                .unwrap();

            self.history.push(bot_response);
            if !tool_calls.is_empty() {
                let call = &tool_calls.first().unwrap();
                println!("Ran {}...", call.name);
                let result = self
                    .tools
                    .run(&call.name, &call.args)
                    .unwrap_or_else(|err| err.to_string());
                println!("Tool returned: {}", result);
                self.push_tool_result(&call.id, result);
            } else {
                println!();
                break;
            }
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
