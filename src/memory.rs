use std::sync::Arc;

use anyhow::Result;
use rusqlite::{Connection, ffi::sqlite3_auto_extension, params};
use serde::Deserialize;
use sqlite_vec::sqlite3_vec_init;

use crate::embedding::Embedder;

const MAX_DISTANCE: f64 = 1.0;

#[derive(Debug, Deserialize)]
pub struct MemoryOp {
    pub op: String,
    pub content: String,
}

pub struct Memory {
    embedder: Arc<Embedder>,
    conn: Connection,
}

impl Memory {
    pub fn new(embedder: &Arc<Embedder>, file: String) -> Result<Self> {
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
        }

        let conn = Connection::open(file)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT
        )",
            [],
        )?;
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings USING vec0(
            mem_id INTEGER PRIMARY KEY,
            embedding float[768]
        )",
            [],
        )?;

        Ok(Memory {
            embedder: embedder.clone(),
            conn,
        })
    }

    pub fn insert(&mut self, msg: &String) -> Result<()> {
        let vector = self.embedder.embed(&msg)?;
        self.conn
            .execute("INSERT INTO memories (text) VALUES (?)", params![msg])?;
        let mem_id = self.conn.last_insert_rowid();
        self.conn.execute(
            "INSERT INTO memory_embeddings (mem_id, embedding) VALUES (?, ?)",
            params![mem_id, as_bytes(&vector)],
        )?;
        Ok(())
    }

    pub fn get_context(&mut self, query: &String) -> Result<Vec<String>> {
        let search_vector = self.embedder.embed(query)?;

        // Query for the top 25 closest matches within the distance cutoff
        let mut stmt = self.conn.prepare(
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

        let mut stmt = self
            .conn
            .prepare("SELECT text FROM memories WHERE id = ?")?;
        let mem_strings = mem_ids
            .iter()
            .map(|id| stmt.query_row(params![id], |row| row.get::<_, String>(0)))
            .collect::<Result<Vec<String>, _>>()?;

        Ok(mem_strings)
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
