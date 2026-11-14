use anyhow::Result;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use tokio::sync::Mutex;

pub struct Embedder {
    model: Mutex<TextEmbedding>,
}

impl Embedder {
    pub fn new() -> Result<Self> {
        let model = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::EmbeddingGemma300M)
                .with_show_download_progress(true),
        )?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    pub async fn embed_document(&self, text: &str) -> Result<Vec<f32>> {
        let formatted = format!("title: memory | text: {text}");
        let mut model = self.model.lock().await;
        let mut embeddings = model.embed(vec![formatted], None)?;
        Ok(embeddings.remove(0))
    }

    pub async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let formatted = format!("task: search result | query: {query}");
        let mut model = self.model.lock().await;
        let mut embeddings = model.embed(vec![formatted], None)?;
        Ok(embeddings.remove(0))
    }
}
