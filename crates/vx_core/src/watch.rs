use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};

use notify::{EventKind, RecursiveMode, Watcher};

/// New contents of a watched file, read on the watcher's own thread.
#[derive(Debug)]
pub struct FileEvent {
    pub path: PathBuf,
    pub contents: std::io::Result<String>,
}

/// Watches a set of files and delivers their new contents when they change, for shader and config
/// hot reload (engineering rule 9).
///
/// Reads happen on the notify thread, so the main thread only drains a channel and never touches
/// the disk (D9). Parent directories are watched rather than the files, because editors often save
/// by writing a temporary file and renaming it over the original.
pub struct FileWatcher {
    _watcher: notify::RecommendedWatcher,
    rx: Receiver<FileEvent>,
}

impl FileWatcher {
    pub fn new(files: &[PathBuf]) -> notify::Result<Self> {
        let files: HashSet<PathBuf> = files.iter().map(|f| canonical(f)).collect();
        let dirs: HashSet<PathBuf> = files
            .iter()
            .filter_map(|f| f.parent().map(Path::to_path_buf))
            .collect();
        let (tx, rx) = channel();
        let watched = files.clone();
        let mut watcher =
            notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
                let Ok(event) = res else { return };
                if !matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_)) {
                    return;
                }
                for path in event.paths {
                    let path = canonical(&path);
                    if watched.contains(&path) {
                        let contents = std::fs::read_to_string(&path);
                        // A closed receiver means the app is shutting down.
                        let _ = tx.send(FileEvent { path, contents });
                    }
                }
            })?;
        for dir in &dirs {
            watcher.watch(dir, RecursiveMode::NonRecursive)?;
        }
        Ok(Self {
            _watcher: watcher,
            rx,
        })
    }

    /// Drains pending changes without blocking, keeping only the latest event per file.
    pub fn poll(&self) -> Vec<FileEvent> {
        let mut latest: Vec<FileEvent> = Vec::new();
        for event in self.rx.try_iter() {
            latest.retain(|e| e.path != event.path);
            latest.push(event);
        }
        latest
    }

    /// True when `path` names the same file as a path this watcher reports.
    pub fn is(event: &FileEvent, path: &Path) -> bool {
        event.path == canonical(path)
    }
}

fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn reports_new_contents_of_a_watched_file() {
        let dir = std::env::temp_dir().join(format!("vx_watch_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("watched.txt");
        let other = dir.join("ignored.txt");
        std::fs::write(&file, "one").unwrap();
        let watcher = FileWatcher::new(std::slice::from_ref(&file)).unwrap();

        // FSEvents can take a moment to start delivering; keep writing until an event arrives.
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut seen = None;
        while seen.is_none() && Instant::now() < deadline {
            std::fs::write(&other, "noise").unwrap();
            std::fs::write(&file, "two").unwrap();
            std::thread::sleep(Duration::from_millis(100));
            seen = watcher
                .poll()
                .into_iter()
                .find(|e| FileWatcher::is(e, &file));
        }
        let event = seen.expect("no event for the watched file");
        assert_eq!(event.contents.unwrap(), "two");
        assert!(watcher.poll().iter().all(|e| !FileWatcher::is(e, &other)));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
