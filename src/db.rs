//! SQLite library: file rows, incremental scan state, and FTS5 over OCR text.
//!
//! The worker connection is the only writer. The window opens a second
//! read-only connection. External-content FTS5 stays in sync through triggers
//! on `ocr_doc`.

use anyhow::{anyhow, Result};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS roots (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    root_id INTEGER NOT NULL REFERENCES roots(id) ON DELETE CASCADE,
    rel_path TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    hash TEXT,
    width INTEGER,
    height INTEGER,
    exif_json TEXT,
    thumb TEXT,
    ocr_status TEXT NOT NULL DEFAULT 'pending',
    embed_status TEXT NOT NULL DEFAULT 'pending',
    ocr_model TEXT,
    embed_model TEXT,
    present INTEGER NOT NULL DEFAULT 1,
    error TEXT,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS files_present_mtime ON files(present, mtime_ns DESC, id DESC);
CREATE INDEX IF NOT EXISTS files_hash ON files(hash);
CREATE TABLE IF NOT EXISTS ocr_doc (
    file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    body TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts USING fts5(
    body,
    content='ocr_doc',
    content_rowid='file_id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS ocr_ai AFTER INSERT ON ocr_doc BEGIN
    INSERT INTO ocr_fts(rowid, body) VALUES (new.file_id, new.body);
END;
CREATE TRIGGER IF NOT EXISTS ocr_ad AFTER DELETE ON ocr_doc BEGIN
    INSERT INTO ocr_fts(ocr_fts, rowid, body) VALUES ('delete', old.file_id, old.body);
END;
CREATE TRIGGER IF NOT EXISTS ocr_au AFTER UPDATE ON ocr_doc BEGIN
    INSERT INTO ocr_fts(ocr_fts, rowid, body) VALUES ('delete', old.file_id, old.body);
    INSERT INTO ocr_fts(rowid, body) VALUES (new.file_id, new.body);
END;
CREATE TABLE IF NOT EXISTS saved_searches (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    mode TEXT NOT NULL,
    query TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS collection_items (
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    PRIMARY KEY (collection_id, file_id)
);
"#;

#[derive(Clone, Debug)]
pub struct ScanFile {
    pub path: String,
    pub rel: String,
    pub size: i64,
    pub mtime_ns: i64,
}

#[derive(Clone, Debug)]
pub struct FileRow {
    pub id: i64,
    pub path: String,
    pub root_id: i64,
    pub size: i64,
    pub mtime_ns: i64,
    pub hash: Option<String>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub exif_json: Option<String>,
    pub thumb: Option<String>,
    pub ocr_status: String,
    pub embed_status: String,
    pub ocr_model: Option<String>,
    pub embed_model: Option<String>,
    pub present: bool,
    pub error: Option<String>,
    pub rel: String,
}

#[derive(Clone, Debug)]
pub struct ClipFrame {
    pub id: i64,
    pub seconds: f64,
    pub path: String,
    pub rel: String,
    pub thumb: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Item {
    pub id: i64,
    pub path: String,
    pub rel: String,
    pub root_id: i64,
    pub thumb: Option<String>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub mtime_ns: i64,
    /// EXIF capture time in Unix nanoseconds. None uses `mtime_ns`.
    pub taken_ns: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct TextHit {
    pub id: i64,
    pub path: String,
    pub rel: String,
    pub thumb: Option<String>,
    pub snippet: String,
    pub score: f32,
    pub width: Option<i64>,
    pub height: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct Detail {
    pub id: i64,
    pub path: String,
    pub rel: String,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub exif_json: String,
    pub ocr: String,
    pub ocr_status: String,
    pub embed_status: String,
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SavedSearch {
    pub id: i64,
    pub name: String,
    pub mode: String,
    pub query: String,
}

#[derive(Clone, Debug)]
pub struct Collection {
    pub id: i64,
    pub name: String,
    pub count: i64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub images: i64,
    pub frames: i64,
    pub with_text: i64,
    pub embedded: i64,
    pub pending: i64,
}

pub struct Library {
    conn: Connection,
}

impl Library {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(SCHEMA)?;
        conn.pragma_update(None, "user_version", 1)?;
        Ok(Self { conn })
    }

    pub fn open_read(path: &Path) -> Result<Self> {
        let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        let _ = conn.pragma_update(None, "query_only", "ON");
        Ok(Self { conn })
    }

    pub fn add_root(&mut self, path: &str) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO roots(path) VALUES(?1) ON CONFLICT(path) DO NOTHING",
            [path],
        )?;
        let id = self
            .conn
            .query_row("SELECT id FROM roots WHERE path=?1", [path], |row| row.get(0))?;
        Ok(id)
    }

    pub fn roots(&self) -> Result<Vec<(i64, String)>> {
        let mut stmt = self.conn.prepare("SELECT id, path FROM roots ORDER BY path")?;
        let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.collect::<rusqlite::Result<_>>().map_err(Into::into)
    }

    pub fn rescan(&mut self, root_id: i64, files: &[ScanFile]) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute("UPDATE files SET present=0 WHERE root_id=?1", [root_id])?;
        let now = now_secs();
        {
            let mut stmt = tx.prepare(
                "INSERT INTO files (path, root_id, rel_path, size, mtime_ns, present, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, 1, ?6)
                 ON CONFLICT(path) DO UPDATE SET
                   root_id=excluded.root_id,
                   rel_path=excluded.rel_path,
                   present=1,
                   updated_at=excluded.updated_at,
                   size=excluded.size,
                   mtime_ns=excluded.mtime_ns,
                   hash=CASE WHEN files.size=excluded.size AND files.mtime_ns=excluded.mtime_ns THEN files.hash ELSE NULL END,
                   width=CASE WHEN files.size=excluded.size AND files.mtime_ns=excluded.mtime_ns THEN files.width ELSE NULL END,
                   height=CASE WHEN files.size=excluded.size AND files.mtime_ns=excluded.mtime_ns THEN files.height ELSE NULL END,
                   exif_json=CASE WHEN files.size=excluded.size AND files.mtime_ns=excluded.mtime_ns THEN files.exif_json ELSE NULL END,
                   thumb=CASE WHEN files.size=excluded.size AND files.mtime_ns=excluded.mtime_ns THEN files.thumb ELSE NULL END,
                   ocr_status=CASE WHEN files.size=excluded.size AND files.mtime_ns=excluded.mtime_ns THEN files.ocr_status ELSE 'pending' END,
                   embed_status=CASE WHEN files.size=excluded.size AND files.mtime_ns=excluded.mtime_ns THEN files.embed_status ELSE 'pending' END,
                   error=CASE WHEN files.size=excluded.size AND files.mtime_ns=excluded.mtime_ns THEN files.error ELSE NULL END",
            )?;
            for file in files {
                stmt.execute(params![file.path, root_id, file.rel, file.size, file.mtime_ns, now])?;
            }
        }
        tx.execute(
            "DELETE FROM ocr_doc WHERE file_id IN (
                SELECT id FROM files WHERE root_id=?1 AND ocr_status='pending'
             )",
            [root_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn pending(&self, root_id: Option<i64>, ocr: bool, embed: bool) -> Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT id FROM files
             WHERE present=1
               AND (?1 IS NULL OR root_id=?1)
               AND (
                    hash IS NULL OR width IS NULL
                    OR (?2 = 1 AND ocr_status='pending')
                    OR (?3 = 1 AND embed_status='pending')
               )
             ORDER BY mtime_ns DESC, id DESC",
        )?;
        let rows = stmt.query_map(params![root_id, ocr as i64, embed as i64], |row| row.get(0))?;
        rows.collect::<rusqlite::Result<_>>().map_err(Into::into)
    }

    pub fn invalidate_ocr(&mut self, stamp: &str) -> Result<usize> {
        if stamp.is_empty() {
            return Ok(0);
        }
        let n = self.conn.execute(
            "UPDATE files SET ocr_status='pending'
             WHERE present=1 AND ocr_status='done' AND IFNULL(ocr_model, '') != ?1",
            [stamp],
        )?;
        Ok(n)
    }

    pub fn invalidate_embed(&mut self, stamp: &str) -> Result<usize> {
        if stamp.is_empty() {
            return Ok(0);
        }
        let n = self.conn.execute(
            "UPDATE files SET embed_status='pending'
             WHERE present=1 AND embed_status='done' AND IFNULL(embed_model, '') != ?1",
            [stamp],
        )?;
        Ok(n)
    }

    pub fn reset_embed_status(&mut self) -> Result<()> {
        self.conn
            .execute("UPDATE files SET embed_status='pending' WHERE embed_status='done'", [])?;
        Ok(())
    }

    pub fn get(&self, id: i64) -> Result<FileRow> {
        self.conn
            .query_row(
                "SELECT id, path, root_id, size, mtime_ns, hash, width, height, exif_json, thumb,
                        ocr_status, embed_status, ocr_model, embed_model, present, error, rel_path
                 FROM files WHERE id=?1",
                [id],
                map_file,
            )
            .map_err(|err| anyhow!("file {id}: {err}"))
    }

    pub fn id_by_path(&self, path: &str) -> Result<Option<i64>> {
        Ok(self
            .conn
            .query_row("SELECT id FROM files WHERE path=?1 AND present=1", [path], |row| row.get(0))
            .optional()?)
    }

    pub fn absent_hash(&self, hash: &str, except: i64) -> Result<Option<FileRow>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id, path, root_id, size, mtime_ns, hash, width, height, exif_json, thumb,
                        ocr_status, embed_status, ocr_model, embed_model, present, error, rel_path
                 FROM files
                 WHERE present=0 AND hash=?1 AND id!=?2
                 ORDER BY (ocr_status='done') DESC, (embed_status='done') DESC, id ASC
                 LIMIT 1",
                params![hash, except],
                map_file,
            )
            .optional()?)
    }

    pub fn set_decoded(
        &mut self,
        id: i64,
        hash: &str,
        width: i64,
        height: i64,
        exif_json: &str,
        thumb: &str,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE files
             SET hash=?2, width=?3, height=?4, exif_json=?5, thumb=?6, updated_at=?7
             WHERE id=?1",
            params![id, hash, width, height, exif_json, thumb, now_secs()],
        )?;
        Ok(())
    }

    pub fn set_ocr(&mut self, id: i64, body: &str, model: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO ocr_doc(file_id, body) VALUES(?1, ?2)
             ON CONFLICT(file_id) DO UPDATE SET body=excluded.body",
            params![id, body],
        )?;
        self.conn.execute(
            "UPDATE files SET ocr_status='done', ocr_model=?2, error=NULL, updated_at=?3 WHERE id=?1",
            params![id, model, now_secs()],
        )?;
        Ok(())
    }

    pub fn set_embed_done(&mut self, id: i64, model: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE files SET embed_status='done', embed_model=?2, updated_at=?3 WHERE id=?1",
            params![id, model, now_secs()],
        )?;
        Ok(())
    }

    pub fn set_ocr_error(&mut self, id: i64, message: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE files SET ocr_status='error', error=?2, updated_at=?3 WHERE id=?1",
            params![id, message, now_secs()],
        )?;
        Ok(())
    }

    pub fn set_embed_error(&mut self, id: i64, message: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE files SET embed_status='error', error=?2, updated_at=?3 WHERE id=?1",
            params![id, message, now_secs()],
        )?;
        Ok(())
    }

    pub fn mark_failed(&mut self, id: i64, message: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE files
             SET ocr_status=CASE WHEN ocr_status='done' THEN ocr_status ELSE 'error' END,
                 embed_status=CASE WHEN embed_status='done' THEN embed_status ELSE 'error' END,
                 error=?2, updated_at=?3
             WHERE id=?1",
            params![id, message, now_secs()],
        )?;
        Ok(())
    }

    pub fn retry_errors(&mut self) -> Result<usize> {
        let n = self.conn.execute(
            "UPDATE files SET
                ocr_status=CASE WHEN ocr_status='error' THEN 'pending' ELSE ocr_status END,
                embed_status=CASE WHEN embed_status='error' THEN 'pending' ELSE embed_status END,
                error=NULL
             WHERE ocr_status='error' OR embed_status='error'",
            [],
        )?;
        Ok(n)
    }

    pub fn ocr_body(&self, id: i64) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT body FROM ocr_doc WHERE file_id=?1", [id], |row| row.get(0))
            .optional()?)
    }

    pub fn delete_file(&mut self, id: i64) -> Result<()> {
        self.conn.execute("DELETE FROM ocr_doc WHERE file_id=?1", [id])?;
        self.conn.execute("DELETE FROM files WHERE id=?1", [id])?;
        Ok(())
    }

    pub fn absent_ids(&self, root_id: i64) -> Result<Vec<i64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM files WHERE present=0 AND root_id=?1")?;
        let rows = stmt.query_map([root_id], |row| row.get(0))?;
        rows.collect::<rusqlite::Result<_>>().map_err(Into::into)
    }

    pub fn file_ids_for_root(&self, root_id: i64) -> Result<Vec<i64>> {
        let mut stmt = self.conn.prepare("SELECT id FROM files WHERE root_id=?1")?;
        let rows = stmt.query_map([root_id], |row| row.get(0))?;
        rows.collect::<rusqlite::Result<_>>().map_err(Into::into)
    }

    pub fn delete_root(&mut self, id: i64) -> Result<()> {
        self.conn.execute("DELETE FROM roots WHERE id=?1", [id])?;
        Ok(())
    }

    fn map_item(row: &rusqlite::Row<'_>) -> rusqlite::Result<Item> {
        let exif: String = row.get(8)?;
        Ok(Item {
            id: row.get(0)?,
            path: row.get(1)?,
            rel: row.get(2)?,
            root_id: row.get(3)?,
            thumb: row.get(4)?,
            width: row.get(5)?,
            height: row.get(6)?,
            mtime_ns: row.get(7)?,
            taken_ns: crate::exif::taken_ns(&exif),
        })
    }

    pub fn gallery(&self, limit: usize) -> Result<Vec<Item>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        // Sort after the read. Capture time lives in exif_json, and this build
        // does not depend on JSON1. The limit is applied after that order, so
        // a copied photo stays with the day it was taken.
        let mut stmt = self.conn.prepare(
            "SELECT id, path, rel_path, root_id, thumb, width, height, mtime_ns, IFNULL(exif_json, '')
             FROM files WHERE present=1",
        )?;
        let rows = stmt.query_map([], Self::map_item)?;
        let mut items: Vec<Item> = rows.collect::<rusqlite::Result<_>>()?;
        order_by_capture(&mut items);
        items.truncate(limit);
        Ok(items)
    }

    pub fn stats(&self) -> Result<Stats> {
        self.conn
            .query_row(
                "SELECT
                    COUNT(*),
                    COALESCE(SUM(CASE WHEN ocr_status='done' AND EXISTS (
                        SELECT 1 FROM ocr_doc d WHERE d.file_id=files.id AND length(d.body)>0
                    ) THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN embed_status='done' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN hash IS NULL OR ocr_status='pending' OR embed_status='pending' THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN rel_path GLOB '* @ [0-9]*.[0-9]*s' THEN 1 ELSE 0 END), 0)
                 FROM files WHERE present=1",
                [],
                |row| {
                    Ok(Stats {
                        images: row.get(0)?,
                        with_text: row.get(1)?,
                        embedded: row.get(2)?,
                        pending: row.get(3)?,
                        frames: row.get(4)?,
                    })
                },
            )
            .map_err(Into::into)
    }

    pub fn detail(&self, id: i64) -> Result<Option<Detail>> {
        Ok(self
            .conn
            .query_row(
                "SELECT f.id, f.path, f.rel_path, f.width, f.height, IFNULL(f.exif_json, ''),
                        IFNULL(d.body, ''), f.ocr_status, f.embed_status, f.error
                 FROM files f LEFT JOIN ocr_doc d ON d.file_id=f.id
                 WHERE f.id=?1",
                [id],
                |row| {
                    Ok(Detail {
                        id: row.get(0)?,
                        path: row.get(1)?,
                        rel: row.get(2)?,
                        width: row.get(3)?,
                        height: row.get(4)?,
                        exif_json: row.get(5)?,
                        ocr: row.get(6)?,
                        ocr_status: row.get(7)?,
                        embed_status: row.get(8)?,
                        error: row.get(9)?,
                    })
                },
            )
            .optional()?)
    }

    pub fn search_fts(&self, match_query: &str, limit: usize) -> Result<Vec<TextHit>> {
        if match_query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn.prepare(
            "SELECT f.id, f.path, f.rel_path, f.thumb,
                    snippet(ocr_fts, 0, '', '', ' … ', 18),
                    bm25(ocr_fts),
                    f.width, f.height
             FROM ocr_fts
             JOIN files f ON f.id = ocr_fts.rowid
             WHERE ocr_fts MATCH ?1 AND f.present=1
             ORDER BY bm25(ocr_fts)
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![match_query, limit as i64], |row| {
            let bm25: f64 = row.get(5)?;
            Ok(TextHit {
                id: row.get(0)?,
                path: row.get(1)?,
                rel: row.get(2)?,
                thumb: row.get(3)?,
                snippet: row.get(4)?,
                score: (-bm25) as f32,
                width: row.get(6)?,
                height: row.get(7)?,
            })
        })?;
        rows.collect::<rusqlite::Result<_>>().map_err(|err| anyhow!("full-text search: {err}"))
    }

    pub fn save_search(&mut self, name: &str, mode: &str, query: &str) -> Result<()> {
        let name = name.trim();
        if name.is_empty() {
            return Err(anyhow!("a saved search needs a name"));
        }
        self.conn.execute(
            "INSERT INTO saved_searches(name, mode, query) VALUES(?1, ?2, ?3)
             ON CONFLICT(name) DO UPDATE SET mode=excluded.mode, query=excluded.query",
            params![name, mode, query],
        )?;
        Ok(())
    }

    pub fn saved_searches(&self) -> Result<Vec<SavedSearch>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, mode, query FROM saved_searches ORDER BY name")?;
        let rows = stmt.query_map([], |row| {
            Ok(SavedSearch {
                id: row.get(0)?,
                name: row.get(1)?,
                mode: row.get(2)?,
                query: row.get(3)?,
            })
        })?;
        rows.collect::<rusqlite::Result<_>>().map_err(Into::into)
    }

    pub fn delete_saved_search(&mut self, id: i64) -> Result<()> {
        self.conn.execute("DELETE FROM saved_searches WHERE id=?1", [id])?;
        Ok(())
    }

    pub fn add_to_collection(&mut self, name: &str, file_ids: &[i64]) -> Result<i64> {
        let name = name.trim();
        if name.is_empty() {
            return Err(anyhow!("a collection needs a name"));
        }
        self.conn.execute(
            "INSERT INTO collections(name) VALUES(?1) ON CONFLICT(name) DO NOTHING",
            [name],
        )?;
        let id: i64 = self
            .conn
            .query_row("SELECT id FROM collections WHERE name=?1", [name], |row| row.get(0))?;
        for file_id in file_ids {
            self.conn.execute(
                "INSERT INTO collection_items(collection_id, file_id) VALUES(?1, ?2)
                 ON CONFLICT(collection_id, file_id) DO NOTHING",
                params![id, file_id],
            )?;
        }
        Ok(id)
    }

    pub fn collections(&self) -> Result<Vec<Collection>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.name, COUNT(i.file_id)
             FROM collections c
             LEFT JOIN collection_items i ON i.collection_id=c.id
             GROUP BY c.id
             ORDER BY c.name",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(Collection {
                id: row.get(0)?,
                name: row.get(1)?,
                count: row.get(2)?,
            })
        })?;
        rows.collect::<rusqlite::Result<_>>().map_err(Into::into)
    }

    pub fn collection_items(&self, id: i64) -> Result<Vec<Item>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.id, f.path, f.rel_path, f.root_id, f.thumb, f.width, f.height, f.mtime_ns,
                    IFNULL(f.exif_json, '')
             FROM collection_items i
             JOIN files f ON f.id=i.file_id
             WHERE i.collection_id=?1 AND f.present=1",
        )?;
        let rows = stmt.query_map([id], Self::map_item)?;
        let mut items: Vec<Item> = rows.collect::<rusqlite::Result<_>>()?;
        order_by_capture(&mut items);
        Ok(items)
    }

    /// Frames sampled from one video, earliest time first.
    pub fn clip_frames(&self, source: &str) -> Result<Vec<ClipFrame>> {
        let source = source.trim();
        if source.is_empty() {
            return Ok(Vec::new());
        }
        let needle = format!("\"source\":{}", serde_json::to_string(source)?);
        let mut stmt = self.conn.prepare(
            "SELECT id, path, rel_path, thumb, IFNULL(exif_json, '')
             FROM files
             WHERE present=1 AND instr(IFNULL(exif_json, ''), ?1) > 0",
        )?;
        let rows = stmt.query_map([needle], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, String>(4)?,
            ))
        })?;
        let mut frames = Vec::new();
        for row in rows {
            let (id, path, rel, thumb, exif) = row?;
            frames.push(ClipFrame {
                id,
                seconds: frame_seconds(&rel, &exif),
                path,
                rel,
                thumb,
            });
        }
        frames.sort_by(|a, b| a.seconds.total_cmp(&b.seconds).then(a.id.cmp(&b.id)));
        Ok(frames)
    }

    pub fn collection_id(&self, name: &str) -> Result<Option<i64>> {
        Ok(self
            .conn
            .query_row("SELECT id FROM collections WHERE name=?1", [name], |row| row.get(0))
            .optional()?)
    }

    pub fn delete_collection(&mut self, id: i64) -> Result<()> {
        self.conn.execute("DELETE FROM collections WHERE id=?1", [id])?;
        Ok(())
    }

    pub fn remove_from_collection(&mut self, collection_id: i64, file_id: i64) -> Result<usize> {
        let removed = self.conn.execute(
            "DELETE FROM collection_items WHERE collection_id=?1 AND file_id=?2",
            params![collection_id, file_id],
        )?;
        Ok(removed)
    }
}

fn frame_seconds(rel: &str, exif: &str) -> f64 {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(exif) {
        if let Some(text) = value.get("time").and_then(|item| item.as_str()) {
            if let Some(seconds) = parse_seconds(text) {
                return seconds;
            }
        }
    }
    rel.rsplit_once(" @ ")
        .and_then(|(_, time)| parse_seconds(time))
        .unwrap_or(0.0)
}

fn parse_seconds(text: &str) -> Option<f64> {
    let text = text.trim().trim_end_matches('s').trim();
    let seconds: f64 = text.parse().ok()?;
    seconds.is_finite().then_some(seconds)
}

fn map_file(row: &rusqlite::Row<'_>) -> rusqlite::Result<FileRow> {
    let present: i64 = row.get(14)?;
    Ok(FileRow {
        id: row.get(0)?,
        path: row.get(1)?,
        root_id: row.get(2)?,
        size: row.get(3)?,
        mtime_ns: row.get(4)?,
        hash: row.get(5)?,
        width: row.get(6)?,
        height: row.get(7)?,
        exif_json: row.get(8)?,
        thumb: row.get(9)?,
        ocr_status: row.get(10)?,
        embed_status: row.get(11)?,
        ocr_model: row.get(12)?,
        embed_model: row.get(13)?,
        present: present != 0,
        error: row.get(15)?,
        rel: row.get(16)?,
    })
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_secs() as i64)
        .unwrap_or(0)
}

fn order_by_capture(items: &mut [Item]) {
    items.sort_by(|a, b| {
        let a_key = a.taken_ns.unwrap_or(a.mtime_ns);
        let b_key = b.taken_ns.unwrap_or(b.mtime_ns);
        b_key.cmp(&a_key).then(b.id.cmp(&a.id))
    });
}

pub fn scratch_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("semantix-{name}-{}-{nanos}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    fn library() -> (PathBuf, Library) {
        let dir = scratch_dir("db");
        let db = Library::open(&dir.join("library.db")).unwrap();
        (dir, db)
    }

    #[test]
    fn fts_roundtrip_finds_receipt_number() {
        let (dir, mut db) = library();
        let root = db.add_root("/photos").unwrap();
        db.rescan(
            root,
            &[ScanFile {
                path: "/photos/receipt.jpg".into(),
                rel: "receipt.jpg".into(),
                size: 10,
                mtime_ns: 1,
            }],
        )
        .unwrap();
        let id = db.pending(Some(root), true, false).unwrap()[0];
        db.set_ocr(id, "Receipt 1042\nAlice Chen", "stamp").unwrap();
        let hits = db.search_fts("1042", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "/photos/receipt.jpg");
        assert!(!hits[0].snippet.is_empty());
        let by_name = db.search_fts(&crate::search::fts_query("Alice"), 10).unwrap();
        assert_eq!(by_name.len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn unchanged_file_is_not_pending_again() {
        let (dir, mut db) = library();
        let root = db.add_root("/photos").unwrap();
        let file = ScanFile {
            path: "/photos/a.jpg".into(),
            rel: "a.jpg".into(),
            size: 10,
            mtime_ns: 5,
        };
        db.rescan(root, &[file.clone()]).unwrap();
        let id = db.pending(Some(root), true, true).unwrap()[0];
        db.set_decoded(id, "abc", 8, 8, "{}", "ab/abc.jpg").unwrap();
        db.set_ocr(id, "hello", "stamp").unwrap();
        db.set_embed_done(id, "clip").unwrap();
        db.rescan(root, &[file]).unwrap();
        assert!(db.pending(Some(root), true, true).unwrap().is_empty());
        let row = db.get(id).unwrap();
        assert_eq!(row.hash.as_deref(), Some("abc"));
        assert_eq!(row.ocr_status, "done");

        db.rescan(
            root,
            &[ScanFile {
                path: "/photos/a.jpg".into(),
                rel: "a.jpg".into(),
                size: 11,
                mtime_ns: 9,
            }],
        )
        .unwrap();
        let again = db.pending(Some(root), true, true).unwrap();
        assert_eq!(again, vec![id]);
        let row = db.get(id).unwrap();
        assert!(row.hash.is_none());
        assert_eq!(row.ocr_status, "pending");
        assert!(db.ocr_body(id).unwrap().is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn saved_search_and_collection_roundtrip() {
        let (dir, mut db) = library();
        let root = db.add_root("/photos").unwrap();
        db.rescan(
            root,
            &[ScanFile {
                path: "/photos/a.jpg".into(),
                rel: "a.jpg".into(),
                size: 4,
                mtime_ns: 1,
            }],
        )
        .unwrap();
        let id = db.pending(Some(root), false, false).unwrap()[0];
        db.save_search("receipts", "text", "1042").unwrap();
        db.save_search("receipts", "hybrid", "alice").unwrap();
        let searches = db.saved_searches().unwrap();
        assert_eq!(searches.len(), 1);
        assert_eq!(searches[0].query, "alice");
        assert_eq!(searches[0].mode, "hybrid");
        let collection = db.add_to_collection("keepers", &[id]).unwrap();
        db.add_to_collection("keepers", &[id]).unwrap();
        let items = db.collection_items(collection).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].path, "/photos/a.jpg");
        assert_eq!(items[0].rel, "a.jpg");
        assert_eq!(db.collections().unwrap()[0].count, 1);
        db.set_decoded(
            id,
            "h",
            8,
            8,
            r#"{"source":"/videos/clip.mp4","time":"2.000s","orientation":1}"#,
            "t.jpg",
        )
        .unwrap();
        db.rescan(
            root,
            &[
                ScanFile {
                    path: "/photos/a.jpg".into(),
                    rel: "a.jpg".into(),
                    size: 4,
                    mtime_ns: 1,
                },
                ScanFile {
                    path: "/photos/early.jpg".into(),
                    rel: "clip.mp4 @ 0.5s".into(),
                    size: 4,
                    mtime_ns: 1,
                },
            ],
        )
        .unwrap();
        let early = db
            .pending(Some(root), false, false)
            .unwrap()
            .into_iter()
            .find(|file_id| db.get(*file_id).unwrap().path.ends_with("early.jpg"))
            .unwrap();
        db.set_decoded(
            early,
            "e",
            8,
            8,
            r#"{"source":"/videos/clip.mp4","time":"0.400s","orientation":1}"#,
            "e.jpg",
        )
        .unwrap();
        let clip = db.clip_frames("/videos/clip.mp4").unwrap();
        assert_eq!(clip.len(), 2);
        assert_eq!(clip[0].id, early);
        assert!((clip[0].seconds - 0.4).abs() < 0.001, "{}", clip[0].seconds);
        assert!(clip[1].seconds > clip[0].seconds);
        assert!(db.clip_frames("/videos/other.mp4").unwrap().is_empty());
        db.add_to_collection("keepers", &[early]).unwrap();
        assert_eq!(db.collections().unwrap()[0].count, 2);
        assert_eq!(db.remove_from_collection(collection, early).unwrap(), 1);
        assert_eq!(db.collection_items(collection).unwrap().len(), 1);
        db.delete_collection(collection).unwrap();
        assert!(db.collections().unwrap().is_empty());
        let stats = db.stats().unwrap();
        assert_eq!(stats.images, 2);
        assert_eq!(stats.frames, 1);
        assert!(db.gallery(10).unwrap().iter().any(|item| item.root_id == root && item.rel == "a.jpg"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn gallery_orders_by_capture_time_then_mtime() {
        let (dir, mut db) = library();
        let root = db.add_root("/photos").unwrap();
        let copied = crate::exif::taken_ns(r#"{"datetime":"2024:12:01 00:00:00"}"#).unwrap();
        let mid = crate::exif::taken_ns(r#"{"datetime":"2022:01:01 00:00:00"}"#).unwrap();
        db.rescan(
            root,
            &[
                ScanFile {
                    path: "/photos/old-copy.jpg".into(),
                    rel: "old-copy.jpg".into(),
                    size: 3,
                    mtime_ns: copied,
                },
                ScanFile {
                    path: "/photos/shot.jpg".into(),
                    rel: "shot.jpg".into(),
                    size: 3,
                    mtime_ns: 1,
                },
                ScanFile {
                    path: "/photos/screen.jpg".into(),
                    rel: "screen.jpg".into(),
                    size: 3,
                    mtime_ns: mid,
                },
            ],
        )
        .unwrap();
        for id in db.pending(Some(root), false, false).unwrap() {
            let row = db.get(id).unwrap();
            let json = if row.rel == "old-copy.jpg" {
                r#"{"datetime":"2020:01:01 00:00:00","orientation":1}"#
            } else if row.rel == "shot.jpg" {
                r#"{"datetime":"2024:06:01 12:00:00","orientation":1}"#
            } else {
                r#"{"orientation":1}"#
            };
            db.set_decoded(id, &row.rel, 8, 8, json, "t.jpg").unwrap();
        }
        let order: Vec<_> = db.gallery(10).unwrap().into_iter().map(|item| item.rel).collect();
        assert_eq!(order, vec!["shot.jpg", "screen.jpg", "old-copy.jpg"]);
        let shot = db.gallery(10).unwrap().into_iter().find(|item| item.rel == "shot.jpg").unwrap();
        assert!(shot.taken_ns.is_some());
        let screen = db.gallery(10).unwrap().into_iter().find(|item| item.rel == "screen.jpg").unwrap();
        assert!(screen.taken_ns.is_none());
        let collection = db.add_to_collection("roll", &db.file_ids_for_root(root).unwrap()).unwrap();
        let collected: Vec<_> = db
            .collection_items(collection)
            .unwrap()
            .into_iter()
            .map(|item| item.rel)
            .collect();
        assert_eq!(collected, order);
        assert_eq!(db.gallery(2).unwrap().len(), 2);
        assert_eq!(db.gallery(0).unwrap().len(), 0);
        let _ = std::fs::remove_dir_all(dir);
    }
}
