use super::memory::Memory;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::pin::Pin;

pub struct ToolManager {
    tools: BTreeMap<String, Tool>,
}

type ToolFuture<'a> = Pin<Box<dyn Future<Output = String> + 'a>>;

struct Tool {
    name: String,
    description: String,
    args: Vec<ToolArgSpec>,
    lambda: Box<dyn for<'a> Fn(&'a HashMap<String, String>, &'a mut Memory) -> ToolFuture<'a>>,
}

struct ToolArgSpec {
    name: &'static str,
    description: &'static str,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCall {
    pub name: String,
    pub args: HashMap<String, String>,
}

impl ToolManager {
    pub fn new() -> Self {
        let search_tool = Tool {
            name: "search-memory".to_string(),
            description: "Search memory for relevant information".to_string(),
            args: vec![ToolArgSpec {
                name: "query",
                description: "Search query to look for",
            }],
            lambda: Box::new(|args, memory| Box::pin(search_memory(args, memory))),
        };

        ToolManager {
            tools: BTreeMap::from([(search_tool.name.clone(), search_tool)]),
        }
    }

    pub fn prompt(&self) -> String {
        let mut prompt = String::from("<tools>");
        for tool in self.tools.values() {
            let args = tool
                .args
                .iter()
                .map(|arg| format!("{}: string", arg.name))
                .collect::<Vec<_>>()
                .join(", ");
            prompt.push_str(&format!(
                "\n- `{}`({}) // {}\n",
                tool.name, args, tool.description
            ));
            prompt.push_str("</tools>\n");
        }
        prompt
    }

    pub fn response_schema(&self) -> Value {
        let mut tool_call_variants: Vec<Value> = self
            .tools
            .values()
            .map(|tool| {
                let mut properties = Map::new();
                let mut required = Vec::new();
                for arg in &tool.args {
                    properties.insert(
                        arg.name.to_string(),
                        json!({
                            "type": "string",
                            "description": arg.description,
                        }),
                    );
                    required.push(Value::String(arg.name.to_string()));
                }

                json!({
                    "type": "object",
                    "description": tool.description.clone(),
                    "properties": {
                        "name": { "enum": [tool.name.clone()] },
                        "args": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                            "additionalProperties": false,
                        },
                    },
                    "required": ["name", "args"],
                    "additionalProperties": false,
                })
            })
            .collect();

        tool_call_variants.push(json!({ "type": "null" }));

        json!({
            "type": "object",
            "properties": {
                "memory_ops": {
                    "type": "array",
                    "description": "Memory operations to perform",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": { "type": "string" },
                            "content": { "type": "string" }
                        },
                        "required": ["op", "content"],
                        "additionalProperties": false
                    },
                    "default": []
                },
                "response": {
                    "type": ["string", "null"],
                    "description": "The final response to the user, or null when calling a tool."
                },
                "tool_call": {
                    "description": "A valid tool call, or null when answering the user.",
                    "anyOf": tool_call_variants,
                },
            },
            "required": ["response", "tool_call", "memory_ops"],
            "additionalProperties": false,
        })
    }

    pub async fn call(&self, call: &ToolCall, memory: &mut Memory) -> String {
        let Some(tool) = self.tools.get(&call.name) else {
            return format!("[FAILED]: unknown tool `{}`", call.name);
        };

        for arg in &tool.args {
            if !call.args.contains_key(arg.name) {
                return format!("[FAILED]: missing `{}` argument", arg.name);
            }
        }

        (tool.lambda)(&call.args, memory).await
    }
}

async fn search_memory(args: &HashMap<String, String>, memory: &mut Memory) -> String {
    let Some(text) = args.get("query") else {
        return "[FAILED]: missing `query` argument".to_string();
    };

    let result = match memory.get_context(text) {
        Ok(res) => format!("[SEARCH]\n{}", res.join("\n")),
        Err(e) => format!("[FAILED] {}", e),
    };

    result
}
