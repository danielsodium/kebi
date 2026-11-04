use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest, ResponseFormat,
        ResponseFormatJsonSchema,
    },
};
use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::de::DeserializeOwned;

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

pub struct Request {
    pub model: String,
    pub messages: Vec<Message>,
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
}

pub enum Response {
    ReasoningDelta { delta: String },
    ContentDelta { delta: String },
}

pub struct StreamResult<T> {
    pub message: Message,
    pub finish_reason: Option<String>,
    pub error: Option<String>,
    pub parse_error: Option<String>,
    pub parsed: Option<T>,
}

fn schema_name<T: JsonSchema>() -> String {
    let name: String = T::schema_name()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();

    if name.is_empty() {
        "structured_output".to_string()
    } else {
        name.chars().take(64).collect()
    }
}

fn normalize_strict(schema: &mut serde_json::Value) {
    match schema {
        serde_json::Value::Object(map) => {
            map.remove("$schema");
            map.remove("title");

            if map.get("type").and_then(|value| value.as_str()) == Some("object") {
                map.insert(
                    "additionalProperties".to_string(),
                    serde_json::Value::Bool(false),
                );

                if let Some(serde_json::Value::Object(properties)) = map.get("properties") {
                    let required = properties
                        .keys()
                        .cloned()
                        .map(serde_json::Value::String)
                        .collect();
                    map.insert("required".to_string(), serde_json::Value::Array(required));
                }
            }

            for (key, value) in map.iter_mut() {
                if key == "properties" {
                    if let serde_json::Value::Object(properties) = value {
                        for property in properties.values_mut() {
                            normalize_strict(property);
                        }
                    }
                } else if key != "required" && key != "additionalProperties" {
                    normalize_strict(value);
                }
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                normalize_strict(item);
            }
        }
        _ => {}
    }
}

fn structured_format<T: JsonSchema>() -> ResponseFormat {
    let schema = schemars::schema_for!(T);
    let mut schema = serde_json::to_value(schema).expect("json schema should serialize");
    normalize_strict(&mut schema);

    ResponseFormat::JsonSchema {
        json_schema: ResponseFormatJsonSchema {
            description: None,
            name: schema_name::<T>(),
            schema,
            strict: Some(true),
        },
    }
}

impl Request {
    fn into_chat_request(self) -> CreateChatCompletionRequest {
        CreateChatCompletionRequest {
            model: self.model,
            messages: self.messages,
            stream: Some(true),
            max_completion_tokens: self.max_tokens,
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

    pub async fn stream<T, F>(
        &self,
        request: Request,
        mut on_event: F,
    ) -> Result<StreamResult<T>, async_openai::error::OpenAIError>
    where
        T: JsonSchema + DeserializeOwned,
        F: FnMut(Response),
    {
        let mut chat_request = request.into_chat_request();
        chat_request.response_format = Some(structured_format::<T>());

        let mut stream = self
            .client
            .chat()
            .create_stream_byot::<_, StreamChunk>(chat_request)
            .await?;

        let mut content = String::new();
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
                    on_event(Response::ContentDelta { delta });
                }
            }
        }

        // Only parse structured output once the server has signalled that the
        // message is complete. A stream that ends early can leave us with a
        // truncated JSON document.
        let complete = error.is_none() && finish_reason.is_some();
        let mut parsed = None;
        let mut parse_error = None;

        if complete {
            match serde_json::from_str::<T>(&content) {
                Ok(value) => parsed = Some(value),
                Err(err) => parse_error = Some(err.to_string()),
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
                    ..Default::default()
                },
            ),
            finish_reason,
            error,
            parse_error,
            parsed,
        })
    }
}
