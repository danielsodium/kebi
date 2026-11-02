use anyhow::Result;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
use walkdir::WalkDir;

#[derive(Deserialize)]
pub struct Pipeline {
    id: String,
    name: String,
    inputs: Vec<String>,
    steps: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RunStatus {
    Running,
    Completed,
    Failed(String),
    Cancelled,
}

pub struct PipelineRun {
    id: Uuid,
    pipeline_id: String,
    status: RunStatus,
    logs: Vec<PipelineMessage>,
    handle: JoinHandle<Result<()>>,
    cancel: CancellationToken,
}

pub struct RunEnv {
    id: Uuid,
    pipeline_id: String,
    variables: HashMap<String, String>,
    files: HashMap<String, String>,
    steps: Vec<String>,
    tx: mpsc::Sender<PipelineUpdate>,
}

pub struct PipelineUpdate {
    id: Uuid,
    pipeline_id: String,
    message: PipelineMessage,
}
pub enum PipelineMessage {
    Debug(String),
    Error(String),
    Status(RunStatus),
    Finish,
}

pub struct Runner {
    pipelines: HashMap<String, Pipeline>,
    runs: HashMap<Uuid, PipelineRun>,
    rx: mpsc::Receiver<PipelineUpdate>,
    tx: mpsc::Sender<PipelineUpdate>,
}

impl Runner {
    pub fn new() -> Result<Self> {
        let home = env::var("HOME")?;
        let tools_dir = PathBuf::from(home).join(".local/share/kebi/pipelines");
        let mut pipelines = HashMap::new();
        for entry in WalkDir::new(&tools_dir).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();

            if !path.is_file() || path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let content = fs::read_to_string(path)?;
            let config: Pipeline = serde_json::from_str(&content)?;
            pipelines.insert(config.id.clone(), config);
        }

        let (tx, rx) = mpsc::channel(100);
        Ok(Runner {
            pipelines,
            runs: HashMap::new(),
            rx,
            tx,
        })
    }

    pub async fn step(&mut self) {
        tokio::select! {
            Some(PipelineUpdate{ id, pipeline_id, message}) = self.rx.recv() => {
                let run: &mut PipelineRun = self.runs.get_mut(&id).unwrap();
                match message {
                    PipelineMessage::Debug(ref msg) => {println!("[{}] ({}) [DEBUG] {}", id, pipeline_id, msg);}
                    PipelineMessage::Error(ref msg) => {println!("[{}] ({}) [ERROR] {}", id, pipeline_id, msg);}
                    PipelineMessage::Status(_) => {println!("[{}] ({}) [STATUS]", id, pipeline_id);}
                    PipelineMessage::Finish => {println!("[{}] ({}) [COMPLETE]", id, pipeline_id);}
                }
                run.logs.push(message);
            }
        }
    }

    pub fn run(&mut self, pipeline_id: String) {
        let cancel = CancellationToken::new();
        let pipeline_steps = self.pipelines.get(&pipeline_id).unwrap().steps.clone();
        let uuid = Uuid::new_v4();
        let run_env = RunEnv {
            id: uuid.clone(),
            pipeline_id: pipeline_id.clone(),
            variables: HashMap::new(),
            files: HashMap::new(),
            steps: pipeline_steps,
            tx: self.tx.clone(),
        };
        let handle = tokio::spawn(async move {
            run_pipeline(run_env).await?;
            Ok(())
        });
        let run = PipelineRun {
            id: uuid,
            status: RunStatus::Running,
            logs: Vec::new(),
            pipeline_id: pipeline_id,
            handle,
            cancel,
        };
        self.runs.insert(run.id.clone(), run);
    }
}

impl RunEnv {
    pub async fn send(&self, msg: PipelineMessage) -> Result<()> {
        self.tx
            .send(PipelineUpdate {
                id: self.id.clone(),
                pipeline_id: self.pipeline_id.clone(),
                message: msg,
            })
            .await?;
        Ok(())
    }
}

async fn run_pipeline(run: RunEnv) -> Result<()> {
    for step in &run.steps {
        let output = Command::new("bash").arg("-c").arg(&step).output().unwrap();

        if output.status.success() {
            let stdout = String::from_utf8(output.stdout).unwrap().trim().to_string();
            if !stdout.is_empty() {
                run.send(PipelineMessage::Debug(stdout)).await?;
            }
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            run.send(PipelineMessage::Error(stderr)).await?;
        }
    }
    run.send(PipelineMessage::Finish).await?;
    Ok(())
}
