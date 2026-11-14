use super::memory::Memory;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::pin::Pin;

pub struct ToolManager {
    tools: BTreeMap<String, Tool>,
}

type ToolFuture<'a> = Pin<Box<dyn Future<Output = String> + Send + 'a>>;

struct Tool {
    name: String,
    description: String,
    args: Vec<ToolArgSpec>,
    lambda: Box<
        dyn for<'a> Fn(&'a HashMap<String, String>, &'a Memory) -> ToolFuture<'a> + Send + Sync,
    >,
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
        let list_mem_tool = Tool {
            name: "list-memories".to_string(),
            description: "List the most recent saved memories when a broad overview is needed or search-memory returns no matches".to_string(),
            args: vec![ToolArgSpec {
                name: "limit",
                description: "Maximum number of memories to return (e.g. '20')",
            }],
            lambda: Box::new(|args, memory| Box::pin(list_memories(args, memory))),
        };
        let api_tool = Tool {
            name: "list-apis".to_string(),
            description: "List the API tools you have access to".to_string(),
            args: vec![ToolArgSpec {
                name: "query",
                description: "Search query for tool to look for",
            }],
            lambda: Box::new(|args, memory| Box::pin(search_tools(args, memory))),
        };

        ToolManager {
            tools: BTreeMap::from([
                (search_tool.name.clone(), search_tool),
                (list_mem_tool.name.clone(), list_mem_tool),
                (api_tool.name.clone(), api_tool),
            ]),
        }
    }

    pub fn prompt(&self) -> String {
        let mut prompt = String::from("<tools>\n");
        for tool in self.tools.values() {
            let args = tool
                .args
                .iter()
                .map(|arg| format!("{}: string ({})", arg.name, arg.description))
                .collect::<Vec<_>>()
                .join(", ");
            prompt.push_str(&format!(
                "- `{}`({}) — {}\n",
                tool.name, args, tool.description
            ));
        }
        prompt.push_str("</tools>\n");
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
                    "description": "Memory operations to save new facts, update outdated facts by id, or delete obsolete facts. Empty array [] if none.",
                    "items": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "op": {
                                        "type": "string",
                                        "enum": ["mem_save"]
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Self-contained atomic sentence about the user with parenthetical domain keywords, e.g. 'User has a severe allergy to peanuts (dietary restriction, food).'"
                                    }
                                },
                                "required": ["op", "content"],
                                "additionalProperties": false
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "op": {
                                        "type": "string",
                                        "enum": ["mem_update"]
                                    },
                                    "id": {
                                        "type": "integer",
                                        "description": "Memory id from [id=N] to overwrite."
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Updated self-contained atomic sentence about the user with parenthetical domain keywords."
                                    }
                                },
                                "required": ["op", "id", "content"],
                                "additionalProperties": false
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "op": {
                                        "type": "string",
                                        "enum": ["mem_delete"]
                                    },
                                    "id": {
                                        "type": "integer",
                                        "description": "Memory id from [id=N] to remove."
                                    }
                                },
                                "required": ["op", "id"],
                                "additionalProperties": false
                            }
                        ]
                    },
                    "default": []
                },
                "tool_call": {
                    "description": "A tool call to execute when information is missing from <retrieved_context>, or null when ready to respond.",
                    "anyOf": tool_call_variants,
                },
                "response": {
                    "type": ["string", "null"],
                    "description": "The concise reply to the user when tool_call is null, or null when calling a tool."
                },
            },
            "required": ["memory_ops", "tool_call", "response"],
            "additionalProperties": false,
        })
    }

    pub async fn call(&self, call: &ToolCall, memory: &Memory) -> String {
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

async fn search_tools(_: &HashMap<String, String>, _: &Memory) -> String {
    "[SEARCH] No tools found".to_string()
}

async fn search_memory(args: &HashMap<String, String>, memory: &Memory) -> String {
    let Some(text) = args.get("query") else {
        return "[FAILED]: missing `query` argument".to_string();
    };

    match memory.get_context(text).await {
        Ok(res) if res.is_empty() => "[SEARCH]\n(No matching memories found)".to_string(),
        Ok(res) => {
            let formatted: Vec<String> = res.iter().map(|m| m.format()).collect();
            format!("[SEARCH]\n- {}", formatted.join("\n- "))
        }
        Err(e) => format!("[FAILED] {}", e),
    }
}

async fn list_memories(args: &HashMap<String, String>, memory: &Memory) -> String {
    let limit = args
        .get("limit")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(20)
        .clamp(1, 100);

    match memory.list_recent(limit).await {
        Ok(res) if res.is_empty() => "[MEMORIES]\n(No saved memories found)".to_string(),
        Ok(res) => {
            let formatted: Vec<String> = res.iter().map(|m| m.format()).collect();
            format!("[MEMORIES]\n- {}", formatted.join("\n- "))
        }
        Err(e) => format!("[FAILED] {}", e),
    }
}
