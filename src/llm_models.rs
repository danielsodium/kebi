use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionMessageToolCall, ChatCompletionMessageToolCallChunk,
        ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
        ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestToolMessage, ChatCompletionRequestToolMessageContent,
        ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
        ChatCompletionTool, ChatCompletionTools, CreateChatCompletionRequest, FunctionCall,
        FunctionObject,
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

#[derive(Clone)]
pub struct ToolFunction {
    pub name: String,
    pub desc: String,
    pub args: Vec<ToolArg>,
}

pub type Message = ChatCompletionRequestMessage;

pub fn system_message(content: String) -> Message {
    ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
        content: ChatCompletionRequestSystemMessageContent::Text(content),
        name: None,
    })
}

pub fn user_message(content: String) -> Message {
    ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
        content: ChatCompletionRequestUserMessageContent::Text(content),
        name: None,
    })
}

pub fn tool_message(tool_call_id: String, content: String) -> Message {
    ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage {
        content: ChatCompletionRequestToolMessageContent::Text(content),
        tool_call_id,
    })
}

pub struct Request {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolFunction>,
    pub max_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct StreamDelta {
    content: Option<String>,
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
}

#[derive(Default)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
}

pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: HashMap<String, String>,
    pub raw_arguments: String,
}

pub enum Response {
    ReasoningDelta { delta: String },
    AnswerDelta { delta: String },
    ToolCall { tool: ToolCall },
}

pub struct StreamResult {
    pub message: Message,
    pub finish_reason: Option<String>,
    pub error: Option<String>,
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
        CreateChatCompletionRequest {
            model: self.model,
            messages: self.messages,
            tools: if self.tools.is_empty() {
                None
            } else {
                Some(self.tools.into_iter().map(to_chat_tool).collect())
            },
            stream: Some(true),
            max_completion_tokens: self.max_tokens,
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
        Self {
            client: Client::with_config(OpenAIConfig::new().with_api_base(url)),
        }
    }

    pub async fn stream<F>(
        &self,
        request: Request,
        mut on_event: F,
    ) -> Result<StreamResult, async_openai::error::OpenAIError>
    where
        F: FnMut(Response),
    {
        let mut stream = self
            .client
            .chat()
            .create_stream_byot::<_, StreamChunk>(request.into_chat_request())
            .await?;

        let mut content = String::new();
        let mut tool_calls: BTreeMap<u32, PendingToolCall> = BTreeMap::new();
        let mut finish_reason: Option<String> = None;
        let mut error: Option<String> = None;

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(err) => {
                    error = Some(err.to_string());
                    break;
                }
            };

            for choice in chunk.choices {
                let StreamChoice {
                    delta:
                        StreamDelta {
                            content: content_delta,
                            reasoning_content,
                            tool_calls: calls,
                        },
                    finish_reason: choice_finish_reason,
                } = choice;

                if choice_finish_reason.is_some() {
                    finish_reason = choice_finish_reason;
                }

                if let Some(delta) = reasoning_content {
                    on_event(Response::ReasoningDelta { delta });
                }

                if let Some(delta) = content_delta {
                    content.push_str(&delta);
                    on_event(Response::AnswerDelta { delta });
                }

                if let Some(calls) = calls {
                    for call in calls {
                        let entry = tool_calls.entry(call.index).or_default();

                        if let Some(id) = call.id {
                            entry.id = id;
                        }
                        if let Some(function) = call.function {
                            if let Some(name) = function.name {
                                entry.name = name;
                            }
                            if let Some(arguments) = function.arguments {
                                entry.arguments.push_str(&arguments);
                            }
                        }
                    }
                }
            }
        }

        let mut assistant_tool_calls = Vec::new();

        // Only trust tool calls once the server has signalled that the message is
        // complete. A stream that ends early can leave us with a partially
        // streamed argument string, and running a tool with truncated args is
        // worse than reporting the incomplete response.
        let complete = error.is_none() && finish_reason.is_some();

        if complete {
            for (_, pending) in tool_calls {
                let PendingToolCall {
                    id,
                    name,
                    arguments: raw_args,
                } = pending;

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

                assistant_tool_calls.push(ChatCompletionMessageToolCalls::Function(
                    ChatCompletionMessageToolCall {
                        id: id.clone(),
                        function: FunctionCall {
                            name: name.clone(),
                            arguments: raw_args.clone(),
                        },
                    },
                ));

                on_event(Response::ToolCall {
                    tool: ToolCall {
                        id,
                        name,
                        args,
                        raw_arguments: raw_args,
                    },
                });
            }
        }

        Ok(StreamResult {
            message: ChatCompletionRequestMessage::Assistant(
                ChatCompletionRequestAssistantMessage {
                    content: if content.is_empty() {
                        None
                    } else {
                        Some(ChatCompletionRequestAssistantMessageContent::Text(content))
                    },
                    tool_calls: if assistant_tool_calls.is_empty() {
                        None
                    } else {
                        Some(assistant_tool_calls)
                    },
                    ..Default::default()
                },
            ),
            finish_reason,
            error,
        })
    }
}
