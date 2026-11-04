use anyhow::Result;
use schemars::JsonSchema;
use serde::Deserialize;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

pub struct ToolManager {
    tools: HashMap<String, Tool>,
}

type ToolFuture<'a> = Pin<Box<dyn Future<Output = String> + 'a>>;

struct Tool {
    name: String,
    description: String,
    args: Vec<String>,
    lambda: Box<dyn for<'a> Fn(&'a HashMap<String, String>) -> ToolFuture<'a>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ToolCall {
    pub name: String,
    pub args: Vec<ToolArg>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ToolArg {
    name: String,
    content: String,
}

impl ToolManager {
    pub fn new() -> Self {
        let fetch_tool = Tool {
            name: "get-request".to_string(),
            description: "Send a GET request to a URL".to_string(),
            args: vec!["url".to_string()],
            lambda: Box::new(|args| Box::pin(fetch(args))),
        };

        ToolManager {
            tools: HashMap::from([(fetch_tool.name.clone(), fetch_tool)]),
        }
    }

    pub fn prompt(&self) -> String {
        let mut tools = String::new();
        for (name, t) in &self.tools {
            let tool_str = format!("- {}({}) : {}", &name, t.args.join(", "), &t.description);
            tools.push_str(&tool_str);
        }
        format!("# Tools\n{}", tools)
    }

    pub async fn call(&self, call: ToolCall) -> String {
        let Some(tool) = self.tools.get(&call.name) else {
            return format!("[FAILED]: unknown tool `{}`", call.name);
        };

        let arg_map = call
            .args
            .into_iter()
            .map(|arg| (arg.name, arg.content))
            .collect();

        (tool.lambda)(&arg_map).await
    }
}

async fn fetch(args: &HashMap<String, String>) -> String {
    let Some(url) = args.get("url") else {
        return "[FAILED]: missing `url` argument".to_string();
    };

    match get_request(url).await {
        Ok(result) => result,
        Err(err) => {
            let err = err.to_string();
            format!("[FAILED]: {err}")
        }
    }
}

async fn get_request(url: &str) -> Result<String> {
    Ok(reqwest::get(url).await?.text().await?)
}
