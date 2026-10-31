use super::llm_models::{
    LlmClient, Message, Request, Response, ToolArg, ToolCall, ToolFunction, system_message,
    tool_message, user_message,
};
use async_recursion::async_recursion;
use std::io::{IsTerminal, Write, stdout};

pub struct Chat {
    model: String,
    tools: Vec<ToolFunction>,
    history: Vec<Message>,
    client: LlmClient,
}

impl Chat {
    pub fn new(url: String, model: String) -> Self {
        Chat {
            model,
            tools: vec![ToolFunction {
                name: "GET request".to_string(),
                desc: "Send a GET request to a URL".to_string(),
                args: vec![ToolArg {
                    name: "URL".to_string(),
                    description: "URL to send a GET Request to".to_string(),
                    options: None,
                }],
                exec: "".to_string(),
            }],
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
            tools: self.tools.clone(),
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

    #[async_recursion]
    pub async fn run(&mut self) {
        let color = std::io::stdout().is_terminal();

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
            println!("Ran {}", &tool_calls.first().unwrap().name);
            self.push_tool_result(&tool_calls.first().unwrap().id, "404 error".to_string());
            self.run().await
        } else {
            println!()
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
