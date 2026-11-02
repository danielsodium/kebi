use super::llm_models::{ToolArg, ToolFunction};
use anyhow::Result;
use std::collections::HashMap;

pub struct ToolRunner {
    pub tools: Vec<ToolFunction>,
}

impl ToolRunner {
    pub fn new() -> Self {
        let tools = vec![
            ToolFunction {
                name: "ls".to_string(),
                desc: "List directories in a path".to_string(),
                args: vec![ToolArg {
                    name: "path".to_string(),
                    description: "relative path".to_string(),
                    options: None,
                }],
            },
            ToolFunction {
                name: "Write file".to_string(),
                desc: "Write a file located at a path".to_string(),
                args: vec![ToolArg {
                    name: "path".to_string(),
                    description: "relative path".to_string(),
                    options: None,
                }],
            },
            ToolFunction {
                name: "Read file".to_string(),
                desc: "Read a file located at a path".to_string(),
                args: vec![ToolArg {
                    name: "path".to_string(),
                    description: "relative path".to_string(),
                    options: None,
                }],
            },
            ToolFunction {
                name: "Read file".to_string(),
                desc: "Read a file located at a path".to_string(),
                args: vec![ToolArg {
                    name: "path".to_string(),
                    description: "relative path".to_string(),
                    options: None,
                }],
            },
        ];
        ToolRunner { tools: tools }
    }

    pub fn run(&self, name: &str, args: &HashMap<String, String>) -> Result<String> {
        Ok("".to_string())
    }
}
