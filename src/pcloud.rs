//! Read a photo through the local pCloud catalog when the drive will not.
//!
//! FUSE reads of files that are not already in the local cache return an I/O
//! error or sit until killed. The same file downloads over HTTPS with the
//! auth token already stored in `~/.pcloud/data.db`. That token stays in this
//! process. It is never written into an error string.

use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

pub(crate) enum Fetch {
    /// This path is not in the local pCloud catalog. The caller may read it
    /// from the filesystem.
    NotCloud,
    Bytes(Vec<u8>),
    /// The catalog has this file and the download did not return it whole.
    Failed,
}

struct Catalog {
    conn: Connection,
    auth: String,
    host: String,
    folders: HashMap<(i64, String), Option<i64>>,
    files: HashMap<(i64, String), Option<(i64, i64)>>,
}

static CATALOG: OnceLock<Mutex<Option<Catalog>>> = OnceLock::new();

pub(crate) fn fetch(path: &Path) -> Fetch {
    let located = {
        let slot = CATALOG.get_or_init(|| Mutex::new(Catalog::open()));
        let Ok(mut guard) = slot.lock() else {
            return Fetch::NotCloud;
        };
        let Some(catalog) = guard.as_mut() else {
            return Fetch::NotCloud;
        };
        match catalog.locate(path) {
            Some((id, size)) => Some((catalog.auth.clone(), catalog.host.clone(), id, size)),
            None => None,
        }
    };
    let Some((auth, host, id, size)) = located else {
        return Fetch::NotCloud;
    };
    match download(&auth, &host, id, size) {
        Some(bytes) => Fetch::Bytes(bytes),
        None => Fetch::Failed,
    }
}

impl Catalog {
    fn open() -> Option<Self> {
        let home = std::env::var_os("HOME")?;
        let path = Path::new(&home).join(".pcloud/data.db");
        if !path.is_file() {
            return None;
        }
        let uri = format!("file:{}?immutable=1", path.display());
        let conn = Connection::open_with_flags(
            uri,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
        )
        .ok()?;
        let auth: String = conn
            .query_row("SELECT value FROM setting WHERE id='auth'", [], |row| row.get(0))
            .ok()?;
        if auth.is_empty() {
            return None;
        }
        let location: String = conn
            .query_row("SELECT value FROM setting WHERE id='location_id'", [], |row| row.get(0))
            .unwrap_or_else(|_| "1".to_string());
        let host = if location == "2" {
            "eapi.pcloud.com"
        } else {
            "api.pcloud.com"
        }
        .to_string();
        Some(Self {
            conn,
            auth,
            host,
            folders: HashMap::new(),
            files: HashMap::new(),
        })
    }

    fn locate(&mut self, path: &Path) -> Option<(i64, i64)> {
        let names: Vec<String> = path
            .components()
            .filter_map(|component| match component {
                std::path::Component::Normal(name) => Some(name.to_string_lossy().into_owned()),
                _ => None,
            })
            .collect();
        if names.len() < 2 {
            return None;
        }
        let last = names.len() - 1;
        for start in 0..last {
            let Some(mut dir) = self.folder(0, &names[start]) else {
                continue;
            };
            let mut ok = true;
            for name in &names[start + 1..last] {
                match self.folder(dir, name) {
                    Some(next) => dir = next,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                if let Some(found) = self.file(dir, &names[last]) {
                    return Some(found);
                }
            }
        }
        None
    }

    fn folder(&mut self, parent: i64, name: &str) -> Option<i64> {
        let key = (parent, name.to_string());
        if let Some(cached) = self.folders.get(&key) {
            return *cached;
        }
        let found = self
            .conn
            .query_row(
                "SELECT id FROM folder WHERE parentfolderid=?1 AND name=?2",
                params![parent, name],
                |row| row.get(0),
            )
            .optional()
            .ok()
            .flatten();
        self.folders.insert(key, found);
        found
    }

    fn file(&mut self, parent: i64, name: &str) -> Option<(i64, i64)> {
        let key = (parent, name.to_string());
        if let Some(cached) = self.files.get(&key) {
            return *cached;
        }
        let found = self
            .conn
            .query_row(
                "SELECT id, IFNULL(size, 0) FROM file WHERE parentfolderid=?1 AND name=?2",
                params![parent, name],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .ok()
            .flatten();
        self.files.insert(key, found);
        found
    }
}

/// Walk `names` until a component is a folder at the pCloud root, then the
/// rest of the path. The mount prefix (`/home/.../pCloudDrive`) is skipped.
#[cfg(test)]
fn locate_in(
    names: &[String],
    mut folder: impl FnMut(i64, &str) -> Option<i64>,
    mut file: impl FnMut(i64, &str) -> Option<(i64, i64)>,
) -> Option<(i64, i64)> {
    let last = names.len() - 1;
    for start in 0..last {
        let Some(mut dir) = folder(0, &names[start]) else {
            continue;
        };
        let mut ok = true;
        for name in &names[start + 1..last] {
            match folder(dir, name) {
                Some(next) => dir = next,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            if let Some(found) = file(dir, &names[last]) {
                return Some(found);
            }
        }
    }
    None
}

fn download(auth: &str, host: &str, file_id: i64, expect: i64) -> Option<Vec<u8>> {
    for _ in 0..2 {
        if let Some(bytes) = download_once(auth, host, file_id, expect) {
            return Some(bytes);
        }
        std::thread::sleep(std::time::Duration::from_millis(400));
    }
    None
}

fn download_once(auth: &str, host: &str, file_id: i64, expect: i64) -> Option<Vec<u8>> {
    if expect > 80 * 1024 * 1024 {
        return None;
    }
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(20))
        .timeout_read(std::time::Duration::from_secs(120))
        .build();
    let link = format!("https://{host}/getfilelink?fileid={file_id}&auth={auth}");
    let response = agent.get(&link).call().ok()?;
    let body = response.into_string().ok()?;
    let value: serde_json::Value = serde_json::from_str(&body).ok()?;
    if value.get("result")?.as_i64()? != 0 {
        return None;
    }
    let cdn_host = value.get("hosts")?.as_array()?.first()?.as_str()?;
    let cdn_path = value.get("path")?.as_str()?;
    let file_url = format!("https://{cdn_host}{cdn_path}");
    let response = agent.get(&file_url).call().ok()?;
    let mut bytes = Vec::new();
    response.into_reader().take(80 * 1024 * 1024).read_to_end(&mut bytes).ok()?;
    if bytes.is_empty() {
        return None;
    }
    if expect > 0 && bytes.len() as i64 != expect {
        return None;
    }
    Some(bytes)
}

#[cfg(test)]
mod tests {
    use super::locate_in;

    fn names(path: &[&str]) -> Vec<String> {
        path.iter().map(|part| (*part).to_string()).collect()
    }

    #[test]
    fn a_mounted_path_resolves_under_the_pcloud_root() {
        let parts = names(&["home", "yy", "pCloudDrive", "bus", "pics", "a.jpg"]);
        let found = locate_in(
            &parts,
            |parent, name| match (parent, name) {
                (0, "bus") => Some(10),
                (10, "pics") => Some(11),
                _ => None,
            },
            |parent, name| {
                if parent == 11 && name == "a.jpg" {
                    Some((99, 5))
                } else {
                    None
                }
            },
        );
        assert_eq!(found, Some((99, 5)));
    }

    #[test]
    fn a_local_path_is_not_a_pcloud_file() {
        let parts = names(&["tmp", "photo.jpg"]);
        let found = locate_in(&parts, |_, _| None, |_, _| None);
        assert_eq!(found, None);
    }
}
