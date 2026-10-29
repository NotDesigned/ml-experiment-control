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
    io::{self, Read, Seek, SeekFrom, Write},
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
    Parts {
        url: String,
        token: String,
        http: reqwest::blocking::Client,
    },
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
            Source::DatasetParts { channel } => {
                Self::parts(channel.url.clone(), channel.token.clone())
            }
            Source::ExchangeParts { channel } => Self::parts(
                format!("{}/datasets/{}", channel.url, config.definition.asset_id()),
                channel.token.clone(),
            ),
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
    fn parts(url: String, token: String) -> Result<Self> {
        Ok(Self::Parts {
            url,
            token,
            http: reqwest::blocking::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .retry(reqwest::retry::never())
                .connect_timeout(Duration::from_secs(5))
                .build()
                .map_err(|_| failed("HTTPS_UNAVAILABLE"))?,
        })
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
            Self::Parts { url, token, http } => {
                if offset % datasets::PART_BYTES != 0
                    || count as u64 != datasets::PART_BYTES.min(total - offset)
                {
                    return Err(failed("PART_RANGE_INVALID"));
                }
                let remaining = deadline
                    .checked_duration_since(Instant::now())
                    .ok_or_else(|| failed("DOWNLOAD_TIMEOUT"))?
                    .min(Duration::from_secs(120));
                let response = http
                    .post(format!("{url}/{}", offset / datasets::PART_BYTES))
                    .bearer_auth(token)
                    .header(ml_exp_core::CLIENT_PROTOCOL_HEADER, "2")
                    .timeout(remaining)
                    .body("{}")
                    .send()
                    .map_err(|_| failed("DATA_TICKET_UNAVAILABLE"))?;
                if response.status() != 200 {
                    return Err(Error {
                        code: "DATA_TICKET_REJECTED",
                        http_status: Some(response.status().as_u16()),
                        errno: None,
                    });
                }
                let mut metadata = Vec::new();
                response
                    .take(32769)
                    .read_to_end(&mut metadata)
                    .map_err(|_| failed("DATA_TICKET_UNAVAILABLE"))?;
                if metadata.len() > 32768 {
                    return Err(failed("DATA_TICKET_INVALID"));
                }
                let ticket = serde_json::Value::Object(
                    ml_exp_core::json::mapping(&metadata)
                        .map_err(|_| failed("DATA_TICKET_INVALID"))?,
                );
                let uri = reqwest::Url::parse(
                    ticket["url"]
                        .as_str()
                        .ok_or_else(|| failed("DATA_TICKET_INVALID"))?,
                )
                .map_err(|_| failed("DATA_TICKET_INVALID"))?;
                if ticket["part"] != offset / datasets::PART_BYTES
                    || ticket["bytes"] != count as u64
                    || uri.scheme() != "https"
                    || uri.port_or_known_default() != Some(443)
                    || !uri.host_str().is_some_and(|h| h.ends_with(".aliyuncs.com"))
                    || !uri.username().is_empty()
                    || uri.password().is_some()
                    || uri.fragment().is_some()
                {
                    return Err(failed("DATA_TICKET_INVALID"));
                }
                // Object download gets no platform credential and no redirects.
                let response = http
                    .get(uri)
                    .timeout(remaining)
                    .send()
                    .map_err(|_| failed("HTTPS_READ_FAILED"))?;
                if response.status() != 200 || response.content_length() != Some(count as u64) {
                    return Err(Error {
                        code: "OBJECT_PART_UNAVAILABLE",
                        http_status: Some(response.status().as_u16()),
                        errno: None,
                    });
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
/// Build-machine staging only: stream an immutable archive to a new owned
/// file, then verify the complete archive and every declared member. The API
/// host never receives this file and no dataset tree is materialized here.
pub fn bundle(config: &Config, output: &Path, cancelled: &AtomicBool) -> Result<()> {
    config
        .validate()
        .map_err(|_| failed("IMPORT_CONFIGURATION_INVALID"))?;
    let parent = Directory::open(
        output
            .parent()
            .ok_or_else(|| failed("SOURCE_PATH_INVALID"))?,
        false,
    )
    .map_err(io_error)?;
    parent.verify_private_parent().map_err(io_error)?;
    let name = output
        .file_name()
        .ok_or_else(|| failed("SOURCE_PATH_INVALID"))?;
    if parent.entry_exists(name).map_err(io_error)? {
        return Err(failed("BUNDLE_DESTINATION_EXISTS"));
    }
    let mut random = [0u8; 16];
    getrandom::fill(&mut random).map_err(|_| failed("BUNDLE_STORAGE_UNAVAILABLE"))?;
    let temporary = std::ffi::OsString::from(format!(
        ".native-data-{:032x}.partial",
        u128::from_be_bytes(random)
    ));
    if parent.capacity().map_err(io_error)?.available_bytes
        < config
            .definition
            .archive_bytes
            .checked_add(1024 * 1024 * 1024)
            .ok_or_else(|| failed("BUNDLE_STORAGE_INSUFFICIENT"))?
    {
        return Err(failed("BUNDLE_STORAGE_INSUFFICIENT"));
    }
    let mut file = parent.create_new_regular(&temporary).map_err(io_error)?;
    let result = (|| {
        let source = Reader::open(config)?;
        let end = Instant::now() + Duration::from_secs(config.timeout_seconds);
        for part in 0..config
            .definition
            .part_count()
            .map_err(|_| failed("IMPORT_CONFIGURATION_INVALID"))?
        {
            let bytes = source.part(
                part * datasets::PART_BYTES,
                config
                    .definition
                    .part_size(part)
                    .map_err(|_| failed("IMPORT_CONFIGURATION_INVALID"))? as usize,
                config.definition.archive_bytes,
                end,
                cancelled,
            )?;
            file.write_all(&bytes).map_err(io_error)?;
        }
        file.sync_all().map_err(io_error)?;
        file.seek(SeekFrom::Start(0)).map_err(io_error)?;
        ml_exp_io::artifacts::verify_data_stream(&mut file, &config.definition)
            .map_err(io_error)?;
        parent
            .publish_regular_as(&temporary, name, &file)
            .map_err(io_error)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = parent.remove_staged_regular(&temporary, &file);
    }
    result
}
#[cfg(test)]
mod tests;
