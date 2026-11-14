//! Offline migration of a record journal from the old layout, which repeats the
//! scope text in every row, to the keyed and compact layout. An operator runs
//! it during a maintenance window with the service stopped.
//!
//! 1. It holds the service's fence and leader locks: a running service makes it
//!    refuse, and a service started meanwhile waits until it ends.
//! 2. It reads the old file without changing it and writes a new file beside
//!    it, streaming rows in key order and committing in bounded batches. Unique
//!    indexes are built after the rows, so the file needs no VACUUM.
//! 3. It verifies the new file against the old one: per scope the row count,
//!    the highest sequence and a SHA256 over every row as readers see it, and
//!    every other table row by row.
//! 4. Only then does it keep the old file under a `pre-compact-<UTC>` name (a
//!    hard link, never deleted) and rename the new file over `outbox.sqlite`,
//!    so that name holds one complete journal at every moment.
//!
//! A rerun after a crash is safe: an unfinished build is discarded and built
//! again, and a verified one is verified again before the swap completes.
use std::{
    ffi::OsStr,
    fs,
    io,
    os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    time::Instant,
};

use ml_exp_io::directory::{Directory, FileLock, LockOptions};
use rusqlite::{Connection, OpenFlags, Statement, params};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{Error, Result};

use super::layout::{self, Layout, ScopeRow};

const JOURNAL: &str = "outbox.sqlite";
const BUILDING: &str = "outbox.sqlite.compact-building";
const VERIFIED: &str = "outbox.sqlite.compact-verified";
const BACKUP: &str = "outbox.sqlite.pre-compact-";
/// Rows written between two commits of the new file.
const BATCH: u64 = 20_000;
/// Rows between two progress lines.
const REPORT_EVERY: u64 = 250_000;
/// Page cache of each connection: 64 MiB.
const CACHE: &str = "PRAGMA cache_size=-65536";
/// Locks the service holds while it runs, in the registry root or one of its
/// directories. Each is taken exclusively without waiting.
const SERVICE_LOCKS: [&str; 5] = [
    ".run-deletion.lock",
    ".local-publisher.lock",
    ".publisher.lock",
    ".observer.lock",
    ".mlflow-publisher.lock",
];
/// Tables whose rows keep their layout: name, columns, key order.
const PLAIN: [(&str, &str, &str); 7] = [
    ("meta", "key,value", "key"),
    ("runs", "project,run,route,config", "project,run"),
    (
        "scopes",
        "id,project,run,attempt,wandb_id,confirmed,status,error,last_attempt_at,last_confirmed_at,failures,display_version,display_confirmed",
        "id",
    ),
    ("native_publications", "scope,state", "scope"),
    (
        "derived_batches",
        "scope,sha256,submitted_at,first,last,records",
        "scope,sha256",
    ),
    (
        "wandb_rebuilds",
        "scope,ordinal,sha256,names,audit,old_wandb_id,new_wandb_id,requested_at,finished_at,previous_state",
        "scope,ordinal",
    ),
    ("mlflow_publications", "scope,state", "scope"),
];
const TABLE_EXISTS: &str =
    "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?)";
const OLD_EVENTS: &str =
    "SELECT seq,event_id,payload,dedup FROM events WHERE scope=? ORDER BY seq";
const NEW_EVENTS: &str =
    "SELECT seq,event_id,form,payload,dedup FROM events WHERE scope_key=? ORDER BY seq";
const OLD_IDS: &str = "SELECT event_id,digest FROM event_ids WHERE scope=? ORDER BY event_id";
const NEW_IDS: &str = "SELECT event_id,digest FROM event_ids WHERE scope_key=? ORDER BY event_id";
const OLD_DERIVED: &str =
    "SELECT seq,event_id,payload,batch FROM derived_events WHERE scope=? ORDER BY seq";
const NEW_DERIVED: &str =
    "SELECT seq,event_id,payload,batch FROM derived_events WHERE scope_key=? ORDER BY seq";
const ADD_KEY: &str = "INSERT INTO scope_keys(scope_key,scope) VALUES (?,?)";

pub struct Options {
    /// The `registry_root` of the read configuration.
    pub registry_root: PathBuf,
    /// Build and verify the new file, then remove it without a swap.
    pub dry_run: bool,
}

#[derive(Debug)]
pub struct Report {
    /// The journal already had the keyed layout; nothing was done.
    pub already_migrated: bool,
    pub scopes: u64,
    pub events: u64,
    pub old_bytes: u64,
    pub new_bytes: u64,
    /// The old file's new name, after a swap.
    pub backup: Option<PathBuf>,
}

fn refused(message: &str) -> Error {
    Error::Refused(message.into())
}

struct Paths {
    directory: Directory,
    journal: PathBuf,
    building: PathBuf,
    verified: PathBuf,
}

impl Paths {
    fn new(registry_root: &Path) -> Result<Self> {
        let tracking = registry_root.join("tracking");
        Ok(Self {
            directory: Directory::open(&tracking, false)?,
            journal: tracking.join(JOURNAL),
            building: tracking.join(BUILDING),
            verified: tracking.join(VERIFIED),
        })
    }
    fn exists(&self, name: &str) -> Result<bool> {
        Ok(self.directory.entry_exists(OsStr::new(name))?)
    }
    /// Remove one of this tool's own unfinished files and its rollback journal.
    fn discard(&self, name: &str) -> Result<()> {
        for leaf in [name.to_owned(), format!("{name}-journal")] {
            if self.exists(&leaf)? {
                self.directory.remove_staged_regular(OsStr::new(&leaf))?;
            }
        }
        Ok(())
    }
    /// A hot rollback journal or WAL beside a file needs SQLite's own recovery.
    fn require_quiet(&self, name: &str) -> Result<()> {
        for suffix in ["-journal", "-wal", "-shm"] {
            if self.exists(&format!("{name}{suffix}"))? {
                let message = format!(
                    "{name}{suffix} exists; start and stop the service once so SQLite \
                     recovers it, then run the migration again"
                );
                return Err(Error::Refused(message));
            }
        }
        Ok(())
    }
}

/// Migrate the journal of `options.registry_root`, printing progress lines.
pub fn run(options: &Options, progress: &mut dyn FnMut(&str)) -> Result<Report> {
    run_until(options, progress, false)
}

/// `run`, which stops before the swap when `stop_before_swap` is set, as a
/// crash there would.
pub(super) fn run_until(
    options: &Options,
    progress: &mut dyn FnMut(&str),
    stop_before_swap: bool,
) -> Result<Report> {
    let _locks = hold_service_locks(&options.registry_root)?;
    let paths = Paths::new(&options.registry_root)?;
    paths.require_quiet(JOURNAL)?;
    let old = fs::symlink_metadata(&paths.journal)?;
    let source = open(&paths.journal, true)?;
    match layout::layout(&source)? {
        Layout::Legacy => {}
        Layout::Keyed => {
            progress("the journal already has the keyed layout; nothing to do");
            return Ok(Report {
                already_migrated: true,
                scopes: 0,
                events: 0,
                old_bytes: old.len(),
                new_bytes: old.len(),
                backup: None,
            });
        }
        Layout::Empty => return Err(refused("the journal has no tables")),
    }
    // A verified file from an interrupted run is used only if it verifies again.
    if !options.dry_run && paths.exists(VERIFIED)? {
        progress("verifying the new file an interrupted run left");
        match verify(&source, &paths.verified, progress) {
            Ok(counts) => return swap(&paths, &old, counts, progress),
            Err(error) => {
                progress(&format!("it does not verify ({error}); building again"));
                paths.discard(VERIFIED)?;
            }
        }
    }
    paths.discard(BUILDING)?;
    let available = paths.directory.capacity()?.available_bytes;
    let needed = old.len().saturating_mul(6) / 5;
    if available < needed {
        let message = format!("{available} bytes are free; the migration needs {needed}");
        return Err(Error::Refused(message));
    }
    let counts = build(&source, &paths.building, &old, progress)?;
    progress("verifying the new file against the old one");
    if verify(&source, &paths.building, progress)? != counts {
        return Err(refused("verification counted other rows"));
    }
    let new_bytes = fs::metadata(&paths.building)?.len();
    let report = Report {
        already_migrated: false,
        scopes: counts.scopes,
        events: counts.events,
        old_bytes: old.len(),
        new_bytes,
        backup: None,
    };
    if options.dry_run {
        paths.discard(BUILDING)?;
        progress("dry run: the new file verified and was removed; nothing was swapped");
        return Ok(report);
    }
    fs::rename(&paths.building, &paths.verified)?;
    paths.directory.sync()?;
    if stop_before_swap {
        return Ok(report);
    }
    swap(&paths, &old, counts, progress)
}

/// Take every service lock that exists, exclusively and without waiting.
fn hold_service_locks(registry_root: &Path) -> Result<Vec<FileLock>> {
    let mut directories = vec![registry_root.to_path_buf()];
    for entry in fs::read_dir(registry_root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            directories.push(entry.path());
        }
    }
    let mut held = Vec::new();
    for path in directories {
        let directory = Directory::open(&path, false)?;
        for name in SERVICE_LOCKS {
            if !directory.entry_exists(OsStr::new(name))? {
                continue;
            }
            match directory.lock(OsStr::new(name), LockOptions::default()) {
                Ok(lock) => held.push(lock),
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    let path = path.join(name).display().to_string();
                    let message = format!("{path} is held: stop the service first");
                    return Err(Error::Refused(message));
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
    Ok(held)
}

fn open(path: &Path, read_only: bool) -> Result<Connection> {
    let mode = if read_only {
        OpenFlags::SQLITE_OPEN_READ_ONLY
    } else {
        OpenFlags::SQLITE_OPEN_READ_WRITE
    };
    let flags = mode | OpenFlags::SQLITE_OPEN_NO_MUTEX | OpenFlags::SQLITE_OPEN_NOFOLLOW;
    let connection = Connection::open_with_flags(path, flags)?;
    connection.execute_batch(CACHE)?;
    Ok(connection)
}

/// Create `path` empty, private and owned as `like`, before SQLite opens it.
fn create_like(path: &Path, like: &fs::Metadata) -> Result<()> {
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    std::os::unix::fs::fchown(&file, Some(like.uid()), Some(like.gid()))?;
    file.set_permissions(fs::Permissions::from_mode(0o600))?;
    file.sync_all()?;
    Ok(())
}

fn exists(connection: &Connection, table: &str) -> Result<bool> {
    let mut statement = connection.prepare(TABLE_EXISTS)?;
    let found = statement.query_row([table], |row| row.get(0))?;
    Ok(found)
}

/// Every scope with a journal row, in text order, which is also key order.
fn journal_scopes(source: &Connection) -> Result<Vec<String>> {
    let mut parts = vec!["SELECT scope FROM events", "SELECT scope FROM event_ids"];
    if exists(source, "derived_events")? {
        parts.push("SELECT scope FROM derived_events");
    }
    let sql = format!("{} ORDER BY 1", parts.join(" UNION "));
    let mut statement = source.prepare(&sql)?;
    let scopes = statement
        .query_map([], |row| row.get(0))?
        .collect::<std::result::Result<_, _>>()?;
    Ok(scopes)
}

fn count(connection: &Connection, table: &str) -> Result<u64> {
    if !exists(connection, table)? {
        return Ok(0);
    }
    let sql = format!("SELECT COUNT(*) FROM {table}");
    let rows: i64 = connection.query_row(&sql, [], |row| row.get(0))?;
    u64::try_from(rows).map_err(|_| Error::Invalid("row count"))
}

#[derive(Debug, PartialEq, Eq)]
struct Counts {
    scopes: u64,
    events: u64,
    event_ids: u64,
    derived: u64,
}

/// Commits the new file every `BATCH` rows and reports now and then.
struct Batches<'a> {
    target: &'a Connection,
    table: &'static str,
    total: u64,
    done: u64,
    commit_at: u64,
    report_at: u64,
    started: Instant,
}

impl<'a> Batches<'a> {
    fn new(target: &'a Connection, table: &'static str, total: u64) -> Self {
        Self {
            target,
            table,
            total,
            done: 0,
            commit_at: BATCH,
            report_at: REPORT_EVERY,
            started: Instant::now(),
        }
    }
    fn row(&mut self, progress: &mut dyn FnMut(&str)) -> Result<()> {
        self.done += 1;
        if self.done >= self.commit_at {
            self.target.execute_batch("COMMIT; BEGIN")?;
            self.commit_at += BATCH;
        }
        if self.done >= self.report_at || self.done == self.total {
            self.report_at += REPORT_EVERY;
            let (table, done, total) = (self.table, self.done, self.total);
            let seconds = self.started.elapsed().as_secs();
            progress(&format!("{table}: {done}/{total} rows, {seconds} s"));
        }
        Ok(())
    }
}

fn build(
    source: &Connection,
    building: &Path,
    old: &fs::Metadata,
    progress: &mut dyn FnMut(&str),
) -> Result<Counts> {
    create_like(building, old)?;
    let target = open(building, false)?;
    // A crash discards this file, so it needs no journal until it is finished.
    target.execute_batch("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF")?;
    source.execute_batch("BEGIN")?;
    target.execute_batch("BEGIN")?;
    target.execute_batch(layout::TABLES)?;
    for (table, columns, order) in PLAIN {
        copy_plain(source, &target, table, columns, order)?;
    }
    target.execute(layout::MARK_LAYOUT, [])?;
    let scopes = journal_scopes(source)?;
    progress(&format!("{} scopes", scopes.len()));
    let mut keys = Vec::with_capacity(scopes.len());
    for (index, scope) in scopes.iter().enumerate() {
        let key = i64::try_from(index + 1).map_err(|_| Error::Invalid("scope count"))?;
        target.execute(ADD_KEY, params![key, scope])?;
        keys.push(ScopeRow::claim(&target, scope)?);
    }
    let mut batches = Batches::new(&target, "events", count(source, "events")?);
    {
        let mut select = source.prepare(OLD_EVENTS)?;
        let mut insert = target.prepare("INSERT INTO events VALUES (?,?,?,?,?,?)")?;
        for (scope, keyed) in scopes.iter().zip(keys.iter_mut()) {
            let mut rows = select.query([scope])?;
            while let Some(row) = rows.next()? {
                let seq: i64 = row.get(0)?;
                let event_id: String = row.get(1)?;
                let payload: String = row.get(2)?;
                let dedup: Option<String> = row.get(3)?;
                let dedup = dedup.as_deref().map(layout::digest_bytes).transpose()?;
                // Text that is not JSON stays exactly as it was.
                let (form, stored) = match serde_json::from_str::<Value>(&payload) {
                    Ok(value) => keyed.stored(&target, &value, payload)?,
                    Err(_) => (None, payload),
                };
                insert.execute(params![keyed.key, seq, event_id, form, stored, dedup])?;
                batches.row(progress)?;
            }
        }
    }
    let events = batches.done;
    let mut batches = Batches::new(&target, "event_ids", count(source, "event_ids")?);
    {
        let mut select = source.prepare(OLD_IDS)?;
        let mut insert = target.prepare("INSERT INTO event_ids VALUES (?,?,?)")?;
        for (scope, keyed) in scopes.iter().zip(&keys) {
            let mut rows = select.query([scope])?;
            while let Some(row) = rows.next()? {
                let event_id: String = row.get(0)?;
                let digest: String = row.get(1)?;
                let digest = layout::digest_bytes(&digest)?;
                insert.execute(params![keyed.key, event_id, digest])?;
                batches.row(progress)?;
            }
        }
    }
    let event_ids = batches.done;
    let total = count(source, "derived_events")?;
    let mut batches = Batches::new(&target, "derived_events", total);
    if total > 0 {
        let mut select = source.prepare(OLD_DERIVED)?;
        let mut insert = target.prepare("INSERT INTO derived_events VALUES (?,?,?,?,?)")?;
        for (scope, keyed) in scopes.iter().zip(&keys) {
            let mut rows = select.query([scope])?;
            while let Some(row) = rows.next()? {
                let seq: i64 = row.get(0)?;
                let event_id: String = row.get(1)?;
                let payload: String = row.get(2)?;
                let batch: String = row.get(3)?;
                insert.execute(params![keyed.key, seq, event_id, payload, batch])?;
                batches.row(progress)?;
            }
        }
    }
    let derived = batches.done;
    progress("building unique indexes");
    target.execute_batch(layout::INDEXES)?;
    target.execute_batch("COMMIT")?;
    source.execute_batch("COMMIT")?;
    progress("checking the new file's integrity");
    let integrity: String = target.query_row("PRAGMA integrity_check", [], |row| row.get(0))?;
    if integrity != "ok" {
        return Err(Error::Refused(format!("new file integrity: {integrity}")));
    }
    target.execute_batch("PRAGMA journal_mode=DELETE")?;
    target.close().map_err(|(_, error)| Error::Sqlite(error))?;
    fs::File::open(building)?.sync_all()?;
    Ok(Counts {
        scopes: scopes.len() as u64,
        events,
        event_ids,
        derived,
    })
}

fn copy_plain(
    source: &Connection,
    target: &Connection,
    table: &str,
    columns: &str,
    order: &str,
) -> Result<()> {
    if !exists(source, table)? {
        return Ok(());
    }
    let width = columns.split(',').count();
    let marks = vec!["?"; width].join(",");
    let sql = format!("INSERT INTO {table}({columns}) VALUES ({marks})");
    let mut insert = target.prepare(&sql)?;
    let sql = format!("SELECT {columns} FROM {table} ORDER BY {order}");
    let mut select = source.prepare(&sql)?;
    let mut rows = select.query([])?;
    while let Some(row) = rows.next()? {
        let mut values = Vec::with_capacity(width);
        for index in 0..width {
            values.push(row.get::<_, rusqlite::types::Value>(index)?);
        }
        insert.execute(rusqlite::params_from_iter(values))?;
    }
    Ok(())
}

/// Row count, highest sequence and SHA256 of some rows.
struct Tally {
    rows: u64,
    highest: i64,
    hasher: Sha256,
}

type Summary = (u64, i64, String);

impl Tally {
    fn new() -> Self {
        Self {
            rows: 0,
            highest: 0,
            hasher: Sha256::new(),
        }
    }
    fn add(&mut self, sequence: Option<i64>, fields: &[Option<&[u8]>]) {
        self.rows += 1;
        if let Some(sequence) = sequence {
            self.highest = self.highest.max(sequence);
            self.hasher.update(sequence.to_le_bytes());
        }
        for field in fields {
            match field {
                None => self.hasher.update([0u8]),
                Some(bytes) => {
                    self.hasher.update([1u8]);
                    self.hasher.update((bytes.len() as u64).to_le_bytes());
                    self.hasher.update(bytes);
                }
            }
        }
    }
    fn finish(self) -> Summary {
        let digest = layout::digest_hex(&self.hasher.finalize());
        (self.rows, self.highest, digest)
    }
}

fn mismatch(table: &str, scope: &str) -> Error {
    Error::Refused(format!("verification failed: {table} of scope {scope} differ"))
}

/// Compare the file at `path` with the old journal, as every reader sees both.
fn verify(source: &Connection, path: &Path, progress: &mut dyn FnMut(&str)) -> Result<Counts> {
    let target = open(path, true)?;
    if layout::layout(&target)? != Layout::Keyed {
        return Err(refused("the new file lacks the keyed layout"));
    }
    source.execute_batch("BEGIN")?;
    target.execute_batch("BEGIN")?;
    let result = compare(source, &target, progress);
    source.execute_batch("COMMIT")?;
    target.execute_batch("COMMIT")?;
    result
}

fn compare(
    source: &Connection,
    target: &Connection,
    progress: &mut dyn FnMut(&str),
) -> Result<Counts> {
    for (table, columns, order) in PLAIN {
        let sql = format!("SELECT {columns} FROM {table} ORDER BY {order}");
        let old = match exists(source, table)? {
            true => plain_tally(source, &sql)?,
            false => Tally::new().finish(),
        };
        let filter = match table {
            "meta" => " WHERE key<>'journal_layout'",
            _ => "",
        };
        let sql = format!("SELECT {columns} FROM {table}{filter} ORDER BY {order}");
        if plain_tally(target, &sql)? != old {
            return Err(mismatch(table, "(all)"));
        }
    }
    let scopes = journal_scopes(source)?;
    if count(target, "scope_keys")? != scopes.len() as u64 {
        return Err(refused("verification failed: scope keys differ"));
    }
    let mut old_events = source.prepare(OLD_EVENTS)?;
    let mut new_events = target.prepare(NEW_EVENTS)?;
    let mut old_ids = source.prepare(OLD_IDS)?;
    let mut new_ids = target.prepare(NEW_IDS)?;
    let mut old_derived = match exists(source, "derived_events")? {
        true => Some(source.prepare(OLD_DERIVED)?),
        false => None,
    };
    let mut new_derived = target.prepare(NEW_DERIVED)?;
    let mut counts = Counts {
        scopes: scopes.len() as u64,
        events: 0,
        event_ids: 0,
        derived: 0,
    };
    for (index, scope) in scopes.iter().enumerate() {
        let keyed = ScopeRow::find(target, scope)?;
        if keyed.key.is_none() {
            return Err(mismatch("keys", scope));
        }
        // Events: the text every reader sees, with its dedup key.
        let mut old = Tally::new();
        let mut rows = old_events.query([scope])?;
        while let Some(row) = rows.next()? {
            let event_id: String = row.get(1)?;
            let payload: String = row.get(2)?;
            let dedup: Option<String> = row.get(3)?;
            let fields = [
                Some(event_id.as_bytes()),
                Some(payload.as_bytes()),
                dedup.as_deref().map(str::as_bytes),
            ];
            old.add(Some(row.get(0)?), &fields);
        }
        let mut new = Tally::new();
        let mut rows = new_events.query([keyed.key])?;
        while let Some(row) = rows.next()? {
            let event_id: String = row.get(1)?;
            let payload = keyed.text(row.get(3)?, row.get(2)?)?;
            let dedup: Option<Vec<u8>> = row.get(4)?;
            let dedup = dedup.as_deref().map(layout::digest_hex);
            let fields = [
                Some(event_id.as_bytes()),
                Some(payload.as_bytes()),
                dedup.as_deref().map(str::as_bytes),
            ];
            new.add(Some(row.get(0)?), &fields);
        }
        let old = old.finish();
        if new.finish() != old {
            return Err(mismatch("events", scope));
        }
        counts.events += old.0;
        // Event IDs and the digest each one is bound to.
        let mut old = Tally::new();
        let mut rows = old_ids.query([scope])?;
        while let Some(row) = rows.next()? {
            let event_id: String = row.get(0)?;
            let digest: String = row.get(1)?;
            let fields = [Some(event_id.as_bytes()), Some(digest.as_bytes())];
            old.add(None, &fields);
        }
        let mut new = Tally::new();
        let mut rows = new_ids.query([keyed.key])?;
        while let Some(row) = rows.next()? {
            let event_id: String = row.get(0)?;
            let digest: Vec<u8> = row.get(1)?;
            let digest = layout::digest_hex(&digest);
            let fields = [Some(event_id.as_bytes()), Some(digest.as_bytes())];
            new.add(None, &fields);
        }
        let old = old.finish();
        if new.finish() != old {
            return Err(mismatch("event_ids", scope));
        }
        counts.event_ids += old.0;
        // Derived rows are copied unchanged.
        let old = match old_derived.as_mut() {
            Some(statement) => derived_tally(statement, [scope])?,
            None => Tally::new().finish(),
        };
        if derived_tally(&mut new_derived, [keyed.key])? != old {
            return Err(mismatch("derived_events", scope));
        }
        counts.derived += old.0;
        if (index + 1) % 1000 == 0 {
            let (done, total) = (index + 1, scopes.len());
            progress(&format!("verified {done}/{total} scopes"));
        }
    }
    // No row may sit under a key that names no scope, or be left out.
    for (table, rows) in [
        ("events", counts.events),
        ("event_ids", counts.event_ids),
        ("derived_events", counts.derived),
    ] {
        if count(target, table)? != rows || count(source, table)? != rows {
            return Err(mismatch(table, "(all)"));
        }
    }
    progress(&format!("verified {} scopes", scopes.len()));
    Ok(counts)
}

fn plain_tally(connection: &Connection, sql: &str) -> Result<Summary> {
    let mut statement = connection.prepare(sql)?;
    let width = statement.column_count();
    let mut tally = Tally::new();
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let mut fields = Vec::with_capacity(width);
        for index in 0..width {
            fields.push(typed(row.get_ref(index)?));
        }
        let fields: Vec<Option<&[u8]>> = fields.iter().map(|value| value.as_deref()).collect();
        tally.add(None, &fields);
    }
    Ok(tally.finish())
}

/// One SQLite value as typed bytes; NULL is none.
fn typed(value: rusqlite::types::ValueRef<'_>) -> Option<Vec<u8>> {
    use rusqlite::types::ValueRef;
    let (tag, bytes) = match value {
        ValueRef::Null => return None,
        ValueRef::Integer(value) => (b'i', value.to_le_bytes().to_vec()),
        ValueRef::Real(value) => (b'r', value.to_bits().to_le_bytes().to_vec()),
        ValueRef::Text(value) => (b't', value.to_vec()),
        ValueRef::Blob(value) => (b'b', value.to_vec()),
    };
    let mut typed = vec![tag];
    typed.extend(bytes);
    Some(typed)
}

fn derived_tally<P: rusqlite::Params>(statement: &mut Statement<'_>, scope: P) -> Result<Summary> {
    let mut tally = Tally::new();
    let mut rows = statement.query(scope)?;
    while let Some(row) = rows.next()? {
        let event_id: String = row.get(1)?;
        let payload: String = row.get(2)?;
        let batch: String = row.get(3)?;
        let fields = [
            Some(event_id.as_bytes()),
            Some(payload.as_bytes()),
            Some(batch.as_bytes()),
        ];
        tally.add(Some(row.get(0)?), &fields);
    }
    Ok(tally.finish())
}

/// Keep the old file under a backup name and move the verified one into place.
fn swap(
    paths: &Paths,
    old: &fs::Metadata,
    counts: Counts,
    progress: &mut dyn FnMut(&str),
) -> Result<Report> {
    paths.require_quiet(JOURNAL)?;
    paths.require_quiet(VERIFIED)?;
    let new = fs::symlink_metadata(&paths.verified)?;
    if new.mode() & 0o777 != 0o600 || new.uid() != old.uid() || new.gid() != old.gid() {
        return Err(refused("the new file's owner or mode differs"));
    }
    let backup = match existing_backup(paths, old)? {
        Some(name) => name,
        None => {
            let name = format!("{BACKUP}{}", utc_stamp());
            let directory = &paths.directory;
            directory.link_regular_from(directory, OsStr::new(JOURNAL), OsStr::new(&name))?;
            name
        }
    };
    fs::rename(&paths.verified, &paths.journal)?;
    paths.directory.sync()?;
    let backup = paths.journal.with_file_name(backup);
    let new_bytes = fs::metadata(&paths.journal)?.len();
    progress(&format!("the old journal is kept as {}", backup.display()));
    Ok(Report {
        already_migrated: false,
        scopes: counts.scopes,
        events: counts.events,
        old_bytes: old.len(),
        new_bytes,
        backup: Some(backup),
    })
}

/// A backup name an interrupted swap already gave the old file.
fn existing_backup(paths: &Paths, old: &fs::Metadata) -> Result<Option<String>> {
    for name in paths.directory.entry_names()? {
        let name = name.to_string_lossy().into_owned();
        if !name.starts_with(BACKUP) {
            continue;
        }
        let identity = paths.directory.regular_identity(OsStr::new(&name))?;
        if identity.is_some_and(|id| id.device == old.dev() && id.inode == old.ino()) {
            return Ok(Some(name));
        }
    }
    Ok(None)
}

fn utc_stamp() -> String {
    let now = time::OffsetDateTime::now_utc();
    let (year, month, day) = (now.year(), u8::from(now.month()), now.day());
    let (hour, minute, second) = (now.hour(), now.minute(), now.second());
    format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z")
}
