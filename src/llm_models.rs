use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionMessageToolCallChunk, ChatCompletionRequestAssistantMessage,
        ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
        ChatCompletionTool, ChatCompletionTools, CreateChatCompletionRequest, FunctionObject,
    },
};
use futures::StreamExt;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};

#[allow(dead_code)]
#[derive(Clone)]
pub struct ToolArg {
    pub name: String,
    pub description: String,
    pub options: Option<Vec<String>>,
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct ToolFunction {
    pub name: String,
    pub desc: String,
    pub args: Vec<ToolArg>,
    pub exec: String,
}

pub struct Request {
    pub model: String,
    pub system: String,
    pub user: String,
    pub history: Vec<String>,
    pub tools: Vec<ToolFunction>,
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Deserialize)]
struct StreamDelta {
    content: Option<String>,
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
}

pub struct ToolCall {
    name: String,
    args: HashMap<String, String>,
}

pub enum Response {
    ReasoningDelta { delta: String },
    AnswerDelta { delta: String },
    ToolCall { tool: ToolCall },
}

fn to_chat_tool(tool: ToolFunction) -> ChatCompletionTools {
    let properties: serde_json::Map<String, serde_json::Value> = tool
        .args
        .into_iter()
        .map(|arg| {
            let mut schema = serde_json::json!({
                "type": "string",
                "description": arg.description,
            });

            if let Some(options) = arg.options {
                schema["oneOf"] = serde_json::Value::Array(
                    options
                        .into_iter()
                        .map(|option| serde_json::json!({ "const": option }))
                        .collect(),
                );
            }

            (arg.name, schema)
        })
        .collect();

    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: tool.name,
            description: Some(tool.desc),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": properties,
            })),
            strict: None,
        },
    })
}

impl Request {
    pub fn into_chat_request(self) -> CreateChatCompletionRequest {
        let mut messages = Vec::with_capacity(self.history.len() + 2);

        if !self.system.is_empty() {
            messages.push(ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessage {
                    content: ChatCompletionRequestSystemMessageContent::Text(self.system),
                    name: None,
                },
            ));
        }

        for (i, entry) in self.history.into_iter().enumerate() {
            messages.push(if i % 2 == 0 {
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text(entry),
                    name: None,
                })
            } else {
                ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                    content: Some(ChatCompletionRequestAssistantMessageContent::Text(entry)),
                    ..Default::default()
                })
            });
        }

        messages.push(ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(self.user),
                name: None,
            },
        ));

        CreateChatCompletionRequest {
            model: self.model,
            messages,
            tools: if self.tools.is_empty() {
                None
            } else {
                Some(self.tools.into_iter().map(to_chat_tool).collect())
            },
            stream: Some(true),
            //temperature: Some(0.2),
            //presence_penalty: Some(0.1),
            ..Default::default()
        }
    }
}

pub struct LlmClient {
    client: Client<OpenAIConfig>,
}

impl LlmClient {
    pub fn new(url: String) -> Self {
        let mut config = OpenAIConfig::new();
        config = config.with_api_base(url);
        config = config.with_api_key("sk-strata-hQwQlvpF5xeXCJG8mUQ0J094sHlUSEp0");
        Self {
            client: Client::with_config(config),
        }
    }

    pub async fn stream<F>(
        &self,
        request: Request,
        mut on_event: F,
    ) -> Result<(), async_openai::error::OpenAIError>
    where
        F: FnMut(Response),
    {
        let mut stream = self
            .client
            .chat()
            .create_stream_byot::<_, StreamChunk>(request.into_chat_request())
            .await?;

        let mut tool_calls: BTreeMap<u32, (String, String, String)> = BTreeMap::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;

            for choice in chunk.choices {
                let StreamDelta {
                    content,
                    reasoning_content,
                    tool_calls: calls,
                } = choice.delta;

                if let Some(delta) = reasoning_content {
                    on_event(Response::ReasoningDelta { delta });
                }

                if let Some(delta) = content {
                    on_event(Response::AnswerDelta { delta });
                }

                if let Some(calls) = calls {
                    for call in calls {
                        let entry = tool_calls.entry(call.index).or_default();

                        if let Some(id) = call.id {
                            entry.0 = id;
                        }
                        if let Some(function) = call.function {
                            if let Some(name) = function.name {
                                entry.1 = name;
                            }
                            if let Some(arguments) = function.arguments {
                                entry.2.push_str(&arguments);
                            }
                        }
                    }
                }
            }
        }

        for (_, (_, name, raw_args)) in tool_calls {
            let parsed: HashMap<String, serde_json::Value> =
                serde_json::from_str(&raw_args).unwrap_or_default();

            let args: HashMap<String, String> = parsed
                .into_iter()
                .map(|(key, value)| {
                    let value = match value {
                        serde_json::Value::String(s) => s,
                        other => other.to_string(),
                    };
                    (key, value)
                })
                .collect();

            on_event(Response::ToolCall {
                tool: ToolCall { name, args },
            });
        }

        Ok(())
    }
}
