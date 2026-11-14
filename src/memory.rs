use std::sync::Arc;

use anyhow::Result;
use rusqlite::{Connection, ffi::sqlite3_auto_extension, params};
use serde::Deserialize;
use sqlite_vec::sqlite3_vec_init;
use tokio::sync::Mutex;

use crate::embedding::Embedder;

const MAX_DISTANCE: f64 = 1.0;

#[derive(Debug, Deserialize)]
#[serde(tag = "op")]
pub enum MemoryOp {
    #[serde(rename = "mem_save")]
    Save { content: String },
    #[serde(rename = "mem_update")]
    Update { id: i64, content: String },
    #[serde(rename = "mem_delete")]
    Delete { id: i64 },
}

pub struct MemoryEntry {
    pub id: i64,
    pub text: String,
}

impl MemoryEntry {
    pub fn format(&self) -> String {
        format!("[id={}] {}", self.id, self.text)
    }
}

pub struct Memory {
    embedder: Arc<Embedder>,
    conn: Mutex<Connection>,
}

impl Memory {
    pub fn new(embedder: &Arc<Embedder>, file: String) -> Result<Self> {
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
        }

        let conn = Connection::open(file)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
                mem_id INTEGER PRIMARY KEY,
                embedding float[768]
            );",
        )?;

        Ok(Memory {
            embedder: embedder.clone(),
            conn: Mutex::new(conn),
        })
    }

    pub async fn apply(&self, op: &MemoryOp) -> Result<()> {
        match op {
            MemoryOp::Save { content } => self.insert(content).await,
            MemoryOp::Update { id, content } => self.update(*id, content).await,
            MemoryOp::Delete { id } => self.delete(*id).await,
        }
    }

    pub async fn insert(&self, msg: &str) -> Result<()> {
        let vector = self.embedder.embed_document(msg).await?;
        let conn = self.conn.lock().await;
        conn.execute("INSERT INTO memories (text) VALUES (?)", params![msg])?;
        let mem_id = conn.last_insert_rowid();
        conn.execute(
            "INSERT INTO memory_embeddings (mem_id, embedding) VALUES (?, ?)",
            params![mem_id, as_bytes(&vector)],
        )?;
        Ok(())
    }

    pub async fn update(&self, id: i64, msg: &str) -> Result<()> {
        let vector = self.embedder.embed_document(msg).await?;
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE memories SET text = ? WHERE id = ?",
            params![msg, id],
        )?;
        conn.execute(
            "UPDATE memory_embeddings SET embedding = ? WHERE mem_id = ?",
            params![as_bytes(&vector), id],
        )?;
        Ok(())
    }

    pub async fn delete(&self, id: i64) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute("DELETE FROM memories WHERE id = ?", params![id])?;
        conn.execute(
            "DELETE FROM memory_embeddings WHERE mem_id = ?",
            params![id],
        )?;
        Ok(())
    }

    pub async fn list_recent(&self, limit: usize) -> Result<Vec<MemoryEntry>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare("SELECT id, text FROM memories ORDER BY id DESC LIMIT ?")?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(MemoryEntry {
                id: row.get(0)?,
                text: row.get(1)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub async fn get_context(&self, query: &str) -> Result<Vec<MemoryEntry>> {
        let search_vector = self.embedder.embed_query(query).await?;
        let conn = self.conn.lock().await;

        // Query for the top 25 closest matches within the distance cutoff
        let mut stmt = conn.prepare(
            "WITH knn AS (
             SELECT mem_id, distance
             FROM memory_embeddings
             WHERE embedding MATCH ? AND k = 25
             ORDER BY distance
         )
         SELECT mem_id, distance
         FROM knn
         WHERE distance <= ?
         ORDER BY distance",
        )?;

        // Execute the query passing the search vector as raw bytes
        let rows = stmt.query_map(params![as_bytes(&search_vector), MAX_DISTANCE], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
        })?;

        let mut mem_ids: Vec<i64> = Vec::new();
        for row in rows {
            let (id, _) = row?;
            mem_ids.push(id);
        }

        let mut stmt = conn.prepare("SELECT id, text FROM memories WHERE id = ?")?;
        let mut entries = Vec::with_capacity(mem_ids.len());
        for id in mem_ids {
            if let Ok(entry) = stmt.query_row(params![id], |row| {
                Ok(MemoryEntry {
                    id: row.get(0)?,
                    text: row.get(1)?,
                })
            }) {
                entries.push(entry);
            }
        }

        Ok(entries)
    }
}

fn as_bytes(vec: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            vec.as_ptr() as *const u8,
            vec.len() * std::mem::size_of::<f32>(),
        )
    }
}
