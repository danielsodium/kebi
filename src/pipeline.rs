use anyhow::Result;
use serde::Deserialize;
use std::cell::RefCell;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::{collections::HashMap, thread};
use walkdir::WalkDir;

#[derive(Deserialize)]
pub struct Pipeline {
    id: String,
    name: String,
    inputs: Vec<String>,
    steps: Vec<String>,
}

#[derive(Default)]
pub struct PipelineRun {
    id: usize,
    pipeline_id: String,
    variables: HashMap<String, String>,
    files: HashMap<String, String>,
    steps: Vec<String>,
}

#[derive(Default)]
pub struct Runner {
    pipelines: RefCell<HashMap<String, Pipeline>>,
    runs: HashMap<usize, PipelineRun>,
}

impl Runner {
    pub fn load(&self) -> Result<()> {
        let home = env::var("HOME")?;
        let tools_dir = PathBuf::from(home).join(".local/share/kebi/pipelines");
        for entry in WalkDir::new(&tools_dir).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();

            if !path.is_file() || path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let content = fs::read_to_string(path)?;
            let config: Pipeline = serde_json::from_str(&content)?;
            self.pipelines
                .borrow_mut()
                .insert(config.id.clone(), config);
        }
        Ok(())
    }
    pub async fn run(&self, id: String) {
        let pipeline_steps = self.pipelines.borrow().get(&id).unwrap().steps.clone();
        let run = PipelineRun {
            id: 0,
            pipeline_id: id,
            variables: HashMap::new(),
            files: HashMap::new(),
            steps: pipeline_steps,
        };
        let handle = thread::spawn(move || run_pipeline(run));
        handle.join().unwrap();
    }
}

fn run_pipeline(run: PipelineRun) {
    for step in run.steps {
        let output = Command::new("bash").arg("-c").arg(step).output().unwrap();

        if output.status.success() {
            let stdout = String::from_utf8(output.stdout).unwrap();
            println!("Output:\n{}", stdout.trim());
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("Command failed with error:\n{}", stderr);
        }
    }
}
