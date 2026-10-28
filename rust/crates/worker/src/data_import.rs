//! CPU-side import into the same immutable dataset store used by the Slurm
//! gateway. Payload bytes live only on backend storage, never the API host.
use crate::datasets::{Config as StoreConfig, Store};
use ml_exp_core::datasets::{
    self, Command, Receipt,
    import::{Config, Progress, Source, Stage},
};
use ml_exp_io::directory::Directory;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, Read},
    os::unix::fs::{FileExt, MetadataExt},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("dataset import failed ({code}); HTTP {http_status:?}")]
pub struct Error {
    pub code: &'static str,
    pub http_status: Option<u16>,
    pub errno: Option<i32>,
}
type Result<T> = std::result::Result<T, Error>;
fn failed(code: &'static str) -> Error {
    Error {
        code,
        http_status: None,
        errno: None,
    }
}
fn io_error(error: io::Error) -> Error {
    Error {
        code: if error.raw_os_error() == Some(libc::ENOSPC) {
            "STORAGE_FULL"
        } else {
            "DATA_IO_FAILED"
        },
        http_status: None,
        errno: error.raw_os_error(),
    }
}
enum Reader {
    File {
        parent: Directory,
        name: std::ffi::OsString,
        file: File,
        size: u64,
    },
    Https {
        url: String,
        http: reqwest::blocking::Client,
    },
}
impl Reader {
    fn open(config: &Config) -> Result<Self> {
        match &config.source {
            Source::File { path } => {
                let path = Path::new(path);
                let parent = Directory::open(
                    path.parent().ok_or_else(|| failed("SOURCE_PATH_INVALID"))?,
                    false,
                )
                .map_err(io_error)?;
                let name = path
                    .file_name()
                    .ok_or_else(|| failed("SOURCE_PATH_INVALID"))?
                    .to_owned();
                let file = parent.open_regular_read(&name).map_err(io_error)?;
                let size = file.metadata().map_err(io_error)?.len();
                if size != config.definition.archive_bytes {
                    return Err(failed("SOURCE_SIZE_MISMATCH"));
                }
                Ok(Self::File {
                    parent,
                    name,
                    file,
                    size,
                })
            }
            Source::Https { url, .. } => {
                let http = reqwest::blocking::Client::builder()
                    .no_proxy()
                    .redirect(reqwest::redirect::Policy::none())
                    .retry(reqwest::retry::never())
                    .connect_timeout(Duration::from_secs(5))
                    .build()
                    .map_err(|_| failed("HTTPS_UNAVAILABLE"))?;
                Ok(Self::Https {
                    url: url.clone(),
                    http,
                })
            }
        }
    }
    fn part(
        &self,
        offset: u64,
        count: usize,
        total: u64,
        deadline: Instant,
        cancelled: &AtomicBool,
    ) -> Result<Vec<u8>> {
        if cancelled.load(Ordering::Relaxed) || Instant::now() >= deadline {
            return Err(failed("CANCELLED_OR_TIMEOUT"));
        }
        match self {
            Self::File {
                parent,
                name,
                file,
                size,
            } => {
                let before = file.metadata().map_err(io_error)?;
                if before.len() != *size || *size != total {
                    return Err(failed("SOURCE_CHANGED"));
                }
                let mut bytes = vec![0; count];
                file.read_exact_at(&mut bytes, offset).map_err(io_error)?;
                parent
                    .validate_regular_entry(name, file)
                    .map_err(io_error)?;
                let after = file.metadata().map_err(io_error)?;
                if before.len() != after.len()
                    || before.mtime() != after.mtime()
                    || before.mtime_nsec() != after.mtime_nsec()
                    || before.ctime() != after.ctime()
                    || before.ctime_nsec() != after.ctime_nsec()
                {
                    return Err(failed("SOURCE_CHANGED"));
                }
                Ok(bytes)
            }
            Self::Https { url, http } => {
                let remaining = deadline
                    .checked_duration_since(Instant::now())
                    .ok_or_else(|| failed("DOWNLOAD_TIMEOUT"))?;
                let end = offset
                    .checked_add(count as u64)
                    .and_then(|n| n.checked_sub(1))
                    .ok_or_else(|| failed("RANGE_INVALID"))?;
                let response = http
                    .get(url)
                    .header(reqwest::header::RANGE, format!("bytes={offset}-{end}"))
                    .header(reqwest::header::ACCEPT_ENCODING, "identity")
                    .timeout(remaining.min(Duration::from_secs(60)))
                    .send()
                    .map_err(|_| failed("HTTPS_READ_FAILED"))?;
                let status = response.status().as_u16();
                if status != 206 {
                    return Err(Error {
                        code: "HTTP_RANGE_REJECTED",
                        http_status: Some(status),
                        errno: None,
                    });
                }
                let expected = format!("bytes {offset}-{end}/{total}");
                if response
                    .headers()
                    .get(reqwest::header::CONTENT_RANGE)
                    .and_then(|s| s.to_str().ok())
                    != Some(expected.as_str())
                    || response
                        .headers()
                        .get(reqwest::header::CONTENT_ENCODING)
                        .is_some_and(|v| v != "identity")
                {
                    return Err(failed("HTTP_RANGE_MISMATCH"));
                }
                let mut bytes = Vec::with_capacity(count);
                response
                    .take(count as u64 + 1)
                    .read_to_end(&mut bytes)
                    .map_err(|_| failed("HTTPS_READ_FAILED"))?;
                if bytes.len() != count {
                    return Err(failed("HTTP_SIZE_MISMATCH"));
                }
                Ok(bytes)
            }
        }
    }
}
fn emit(
    config: &Config,
    state: Option<&datasets::State>,
    phase: Stage,
    resumed: u64,
    error: Option<&str>,
    progress: &mut dyn FnMut(&Progress) -> io::Result<()>,
) -> Result<()> {
    let value = config
        .progress(state, phase, resumed, error)
        .map_err(|_| failed("PROGRESS_IDENTITY_INVALID"))?;
    progress(&value).map_err(|_| failed("PROGRESS_UNAVAILABLE"))
}
/// The destination root must be initialized explicitly before entering this
/// function. Missing/corrupt roots are not bootstrapped or repaired on a retry.
pub fn materialize(
    config: &Config,
    cancelled: &AtomicBool,
    progress: &mut dyn FnMut(&Progress) -> io::Result<()>,
) -> Result<Receipt> {
    config
        .validate()
        .map_err(|_| failed("IMPORT_CONFIGURATION_INVALID"))?;
    let deadline = Instant::now() + Duration::from_secs(config.timeout_seconds);
    let store = Store::open(StoreConfig {
        root: config.root.clone().into(),
        target: config.target.clone(),
    })
    .map_err(io_error)?;
    let request = config
        .request(Command::Begin {
            definition: config.definition.clone(),
        })
        .map_err(|_| failed("IMPORT_CONFIGURATION_INVALID"))?;
    let mut state = store
        .handle(&request, &[], cancelled)
        .map_err(io_error)?
        .state;
    let resumed = state.parts.len() as u64;
    let run = |state: &mut datasets::State,
               progress: &mut dyn FnMut(&Progress) -> io::Result<()>|
     -> Result<Receipt> {
        emit(
            config,
            Some(state),
            Stage::Initializing,
            resumed,
            None,
            progress,
        )?;
        if state.ready.is_none() {
            let reader = Reader::open(config)?;
            let count = config
                .definition
                .part_count()
                .map_err(|_| failed("PART_IDENTITY_INVALID"))?;
            for number in 0..count {
                if state.parts.contains_key(&number) {
                    continue;
                }
                emit(
                    config,
                    Some(state),
                    Stage::Downloading,
                    resumed,
                    None,
                    progress,
                )?;
                let bytes = config
                    .definition
                    .part_size(number)
                    .map_err(|_| failed("PART_IDENTITY_INVALID"))?;
                let offset = number
                    .checked_mul(datasets::PART_BYTES)
                    .ok_or_else(|| failed("PART_IDENTITY_INVALID"))?;
                let payload = reader.part(
                    offset,
                    bytes as usize,
                    config.definition.archive_bytes,
                    deadline,
                    cancelled,
                )?;
                if cancelled.load(Ordering::Relaxed) || Instant::now() >= deadline {
                    return Err(failed("CANCELLED_OR_TIMEOUT"));
                }
                let request = config
                    .request(Command::Part {
                        asset_id: config.definition.asset_id(),
                        number,
                        sha256: format!("{:x}", Sha256::digest(&payload)),
                        bytes,
                    })
                    .map_err(|_| failed("PART_IDENTITY_INVALID"))?;
                *state = store
                    .handle(&request, &payload, cancelled)
                    .map_err(io_error)?
                    .state;
                emit(
                    config,
                    Some(state),
                    Stage::Downloading,
                    resumed,
                    None,
                    progress,
                )?;
            }
        }
        emit(
            config,
            Some(state),
            Stage::Verifying,
            resumed,
            None,
            progress,
        )?;
        if cancelled.load(Ordering::Relaxed) || Instant::now() >= deadline {
            return Err(failed("CANCELLED_OR_TIMEOUT"));
        }
        let request = config
            .request(Command::Complete {
                asset_id: config.definition.asset_id(),
            })
            .map_err(|_| failed("DATASET_IDENTITY_INVALID"))?;
        *state = store
            .handle(&request, &[], cancelled)
            .map_err(io_error)?
            .state;
        let receipt = state
            .ready
            .clone()
            .ok_or_else(|| failed("READY_RECEIPT_MISSING"))?;
        emit(config, Some(state), Stage::Ready, resumed, None, progress)?;
        Ok(receipt)
    };
    match run(&mut state, progress) {
        Ok(receipt) => Ok(receipt),
        Err(error) => {
            let _ = emit(
                config,
                Some(&state),
                Stage::Failed,
                resumed,
                Some(error.code),
                progress,
            );
            Err(error)
        }
    }
}
pub fn initialize(config: &Config) -> Result<()> {
    config
        .validate()
        .map_err(|_| failed("IMPORT_CONFIGURATION_INVALID"))?;
    Store::initialize(StoreConfig {
        root: config.root.clone().into(),
        target: config.target.clone(),
    })
    .map_err(io_error)
}
pub fn load(path: &Path) -> Result<Config> {
    let bytes =
        ml_exp_io::read_private(path, datasets::MAX_METADATA as u64, true).map_err(io_error)?;
    let config: Config = serde_json::from_value(serde_json::Value::Object(
        ml_exp_core::json::mapping(&bytes).map_err(|_| failed("IMPORT_CONFIGURATION_INVALID"))?,
    ))
    .map_err(|_| failed("IMPORT_CONFIGURATION_INVALID"))?;
    config
        .validate()
        .map_err(|_| failed("IMPORT_CONFIGURATION_INVALID"))?;
    Ok(config)
}
#[cfg(test)]
mod tests;
