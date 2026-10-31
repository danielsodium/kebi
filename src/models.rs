use super::llm_models::{LlmClient, Request, Response, ToolArg, ToolCall, ToolFunction};
use std::io::{IsTerminal, Write, stdout};

pub struct Chat {
    model: String,
    system: Vec<String>,
    tools: Vec<ToolFunction>,
    client: LlmClient,
}

impl Chat {
    pub fn new(url: String, model: String) -> Self {
        Chat {
            model,
            system: vec!["Answer directly and concisely without internal monologue, planning steps, or a chain-of-thought process.".to_string()],
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
            client: LlmClient::new(url)
        }
    }

    fn request(&self, prompt: String) -> Request {
        Request {
            model: self.model.clone(),
            system: self.system.concat(),
            user: prompt,
            history: vec![],
            tools: self.tools.clone(),
        }
    }

    pub async fn start(&self, prompt: String) {
        let color = std::io::stdout().is_terminal();

        let mut reasoning_deltas: Vec<String> = Vec::new();
        let mut answer_deltas: Vec<String> = Vec::new();
        let mut tool_call: Option<ToolCall> = None;

        self.client
            .stream(self.request(prompt), |event| match event {
                Response::ReasoningDelta { delta } => {
                    reasoning_deltas.push(delta);
                    print_delta(color, &reasoning_deltas)
                }
                Response::AnswerDelta { delta } => {
                    answer_deltas.push(delta);
                    print_delta(false, &answer_deltas);
                }
                Response::ToolCall { tool } => tool_call = Some(tool),
            })
            .await
            .unwrap();

        let reasoning = reasoning_deltas.concat();
        let answer = answer_deltas.concat();
        let tool_call = tool_call;

        if tool_call.is_some() {}
    }
}

fn print_delta(color: bool, deltas: &Vec<String>) {
    let delta = &deltas[deltas.len() - 1];
    if color {
        let _ = print!("\x1b[2m{delta}\x1b[0m");
    } else {
        let _ = print!("{delta}");
    }
    stdout().flush().unwrap()
}
