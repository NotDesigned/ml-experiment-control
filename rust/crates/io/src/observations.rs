//! Read only bounded windows and an append-only history page. A concurrent
//! change is an explicit retryable read failure, never fabricated empty data.
use crate::directory::Directory;
use ml_exp_core::{
    managed::{
        history::{self, Cursor, Page},
        remote::{Problem, WINDOW_BYTES, Window},
    },
    slurm::helper,
};
use sha2::{Digest, Sha256};
use std::{
    ffi::OsStr,
    io::{self, Read, Seek, SeekFrom},
    os::unix::fs::MetadataExt,
};
fn changed(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    a.len() != b.len()
        || a.mtime() != b.mtime()
        || a.mtime_nsec() != b.mtime_nsec()
        || a.ctime() != b.ctime()
        || a.ctime_nsec() != b.ctime_nsec()
}
pub fn window(directory: &Directory, name: &str) -> Window {
    let read = || -> io::Result<Window> {
        if !directory.entry_exists(OsStr::new(name))? {
            return Ok(Window::unavailable(Problem::Missing));
        }
        let mut file = directory.open_regular_read(OsStr::new(name))?;
        let before = file.metadata()?;
        let mut offset = before.len().saturating_sub(WINDOW_BYTES as u64);
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = Vec::new();
        (&mut file)
            .take((WINDOW_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        directory.validate_regular_entry(OsStr::new(name), &file)?;
        if bytes.len() as u64 != before.len() - offset || changed(&before, &file.metadata()?) {
            return Ok(Window::unavailable(Problem::Changed));
        }
        if offset > 0 {
            let trim = bytes
                .iter()
                .take(3)
                .take_while(|b| **b & 0xc0 == 0x80)
                .count();
            bytes.drain(..trim);
            offset += trim as u64;
        }
        match String::from_utf8(bytes) {
            Ok(text) => Ok(Window {
                text,
                offset_bytes: offset,
                total_bytes: before.len(),
                problem: (offset > 0).then_some(Problem::Truncated),
            }),
            Err(_) => Ok(Window::unavailable(Problem::InvalidUtf8)),
        }
    };
    read().unwrap_or_else(|_| Window::unavailable(Problem::Unavailable))
}
pub fn history(
    directory: &Directory,
    start: Option<&Cursor>,
    finished: bool,
    config_sha256: &str,
) -> io::Result<Option<Page>> {
    if !directory.entry_exists(OsStr::new("metrics.jsonl"))? {
        if start.is_some() {
            return Err(io::Error::other("metrics file disappeared"));
        }
        return Ok(finished.then(|| Page {
            contract: history::CONTRACT.into(),
            config_sha256: config_sha256.into(),
            start: None,
            next: Cursor {
                device: 0,
                inode: 0,
                offset_bytes: 0,
                prefix_sha256: helper::sha256_bytes(&[]),
            },
            bytes: Vec::new(),
            total_bytes: 0,
            process_finished: true,
            missing_after_finish: true,
        }));
    }
    let mut file = directory.open_regular_read(OsStr::new("metrics.jsonl"))?;
    let before = file.metadata()?;
    let offset = start.map_or(0, |c| c.offset_bytes);
    if before.len() > history::MAX_FILE_BYTES
        || offset > before.len()
        || start.is_some_and(|c| c.device != before.dev() || c.inode != before.ino())
    {
        return Err(io::Error::other("metrics file identity changed"));
    }
    let mut hash = Sha256::new();
    let mut remaining = offset;
    let mut buffer = [0u8; 16384];
    while remaining > 0 {
        let count = remaining.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..count])?;
        hash.update(&buffer[..count]);
        remaining -= count as u64;
    }
    if start.is_some_and(|c| format!("sha256:{:x}", hash.clone().finalize()) != c.prefix_sha256) {
        return Err(io::Error::other("metrics prefix changed"));
    }
    let count = (before.len() - offset).min(history::PAGE_BYTES as u64) as usize;
    let mut bytes = vec![0; count];
    file.read_exact(&mut bytes)?;
    hash.update(&bytes);
    directory.validate_regular_entry(OsStr::new("metrics.jsonl"), &file)?;
    if changed(&before, &file.metadata()?) {
        return Err(io::Error::other("metrics changed while sampling"));
    }
    let page = Page {
        contract: history::CONTRACT.into(),
        config_sha256: config_sha256.into(),
        start: start.cloned(),
        next: Cursor {
            device: before.dev(),
            inode: before.ino(),
            offset_bytes: offset + count as u64,
            prefix_sha256: format!("sha256:{:x}", hash.finalize()),
        },
        bytes,
        total_bytes: before.len(),
        process_finished: finished,
        missing_after_finish: false,
    };
    page.validate(start)
        .map_err(|_| io::Error::other("invalid metrics page"))?;
    Ok(Some(page))
}
