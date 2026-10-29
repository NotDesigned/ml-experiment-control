//! Dataset preparation occurs before the at-most-once training spawn. Its
//! durable cache and progress are independent from the output archive.
use crate::{
    data_import,
    managed::{self, Error, Result},
};
use ml_exp_core::{
    cloud::launch::Package,
    input_progress::{Phase, Record},
};
use ml_exp_io::directory::Directory;
use std::{collections::BTreeMap, ffi::OsStr, sync::atomic::AtomicBool, time::Instant};
const NAME: &str = ".cloud-input-progress.json";
#[cfg(test)]
mod tests;
/// Copy jobs may initialize only their frozen dataset cache's parent beneath
/// an existing mount. Old jobs omit the plan and retain their original behavior.
pub fn scoped_cache(
    config: &ml_exp_core::datasets::import::Config,
    storage: Option<&ml_exp_core::cloud::launch::StorageLayout>,
) -> Result<()> {
    config.validate().map_err(|_| Error::Configuration)?;
    if let Some(storage) = storage {
        if std::path::Path::new(&config.root) != storage.datasets {
            return Err(Error::Binding);
        }
        crate::cloud_storage::prepare_data(storage)?;
    }
    cache(config)
}
pub fn cache(config: &ml_exp_core::datasets::import::Config) -> Result<()> {
    let path = std::path::Path::new(&config.root);
    let parent = Directory::open(path.parent().ok_or(Error::Configuration)?, false)
        .map_err(|_| Error::Storage)?;
    parent.verify_private_parent().map_err(|_| Error::Storage)?;
    let name = path.file_name().ok_or(Error::Configuration)?;
    let store = || {
        crate::datasets::Store::open(crate::datasets::Config {
            root: path.into(),
            target: config.target.clone(),
        })
        .map(|_| ())
        .map_err(|_| Error::Storage)
    };
    if parent.entry_exists(name).map_err(|_| Error::Storage)? {
        return store();
    }
    // Reserve the final name exclusively. Completion is the store marker,
    // not directory visibility; a crash leaves an unavailable, unadopted root.
    // This requires no renameat2 support from the shared filesystem.
    match parent.create_new_child(name) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => return store(),
        Err(_) => return Err(Error::Storage),
    }
    crate::datasets::Store::initialize(crate::datasets::Config {
        root: path.into(),
        target: config.target.clone(),
    })
    .map_err(|_| Error::Storage)?;
    store()
}
fn save(root: &Directory, records: &BTreeMap<String, Record>) -> std::io::Result<()> {
    root.replace_regular(
        OsStr::new(NAME),
        &serde_json::to_vec(&records.values().collect::<Vec<_>>())?,
    )
}
pub fn read(package: &Package) -> Result<Vec<Record>> {
    let root = Directory::open(&package.config.attempt_root, false).map_err(|_| Error::Storage)?;
    if !root
        .entry_exists(OsStr::new(NAME))
        .map_err(|_| Error::Storage)?
    {
        return Ok(Vec::new());
    }
    let bytes = root
        .read_regular(OsStr::new(NAME), 128 * 1024)
        .map_err(|_| Error::Storage)?;
    let value: Vec<Record> = serde_json::from_slice(&bytes).map_err(|_| Error::Binding)?;
    if value.len() > 32 {
        return Err(Error::Binding);
    }
    Ok(value)
}
pub fn materialize(package: &Package, cancelled: &AtomicBool) -> managed::Result<()> {
    let root = Directory::open(&package.config.attempt_root, false).map_err(|_| Error::Storage)?;
    let mut records: BTreeMap<String, Record> = read(package)?
        .into_iter()
        .map(|r| (r.asset_id.clone(), r))
        .collect();
    for config in &package.imports {
        let start = Instant::now();
        let mut baseline = None;
        let result = cache(config)
            .map_err(|_| data_import::Error {
                code: "DATA_CACHE_UNAVAILABLE",
                http_status: None,
                errno: None,
            })
            .and_then(|_| {
                data_import::materialize(config, cancelled, &mut |progress| {
                    let baseline = *baseline.get_or_insert(progress.received_bytes);
                    let elapsed = start.elapsed().as_secs_f64();
                    let record = Record {
                        at: time::OffsetDateTime::now_utc()
                            .format(&time::format_description::well_known::Rfc3339)
                            .map_err(std::io::Error::other)?,
                        phase: progress.stage,
                        asset_id: progress.asset_id.clone(),
                        received_bytes: Some(progress.received_bytes as f64),
                        expected_bytes: Some(progress.total_bytes as f64),
                        elapsed_seconds: Some(elapsed),
                        bytes_per_second: (elapsed > 0.).then(|| {
                            progress.received_bytes.saturating_sub(baseline) as f64 / elapsed
                        }),
                        http_status: None,
                        errno: None,
                        code: progress.error_code.clone(),
                        error_class: None,
                        failed_phase: None,
                    };
                    records.insert(record.asset_id.clone(), record);
                    save(&root, &records)
                })
            });
        match result {
            Ok(receipt) => {
                let expected = package
                    .config
                    .datasets
                    .iter()
                    .find(|m| m.receipt.asset_id == receipt.asset_id)
                    .ok_or(Error::Binding)?;
                if expected.receipt != receipt {
                    return Err(Error::Binding);
                }
            }
            Err(error) => {
                let record = records
                    .entry(config.definition.asset_id())
                    .or_insert_with(|| Record {
                        at: time::OffsetDateTime::now_utc()
                            .format(&time::format_description::well_known::Rfc3339)
                            .unwrap_or_default(),
                        phase: Phase::Failed,
                        asset_id: config.definition.asset_id(),
                        received_bytes: Some(0.),
                        expected_bytes: Some(config.definition.archive_bytes as f64),
                        elapsed_seconds: Some(start.elapsed().as_secs_f64()),
                        bytes_per_second: None,
                        http_status: None,
                        errno: None,
                        code: None,
                        error_class: None,
                        failed_phase: None,
                    });
                record.phase = Phase::Failed;
                record.code = Some(error.code.into());
                record.http_status = error.http_status.map(f64::from);
                record.errno = error.errno.map(f64::from);
                save(&root, &records).map_err(|_| Error::Storage)?;
                return Err(Error::Storage);
            }
        }
    }
    Ok(())
}
