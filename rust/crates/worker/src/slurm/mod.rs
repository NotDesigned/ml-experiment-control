//! Scoped native Slurm metadata operations. No arbitrary shell RPC, data upload,
//! automatic retry, runtime staging or scheduler-ownership verification.
#![forbid(unsafe_code)]
mod ledger;
pub use ledger::{Helper, HelperConfig};

use ml_exp_core::slurm::helper;
pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid trusted Slurm helper configuration")]
    Configuration,
    #[error("Slurm helper metadata storage is unavailable; do not replay")]
    Storage(#[from] std::io::Error),
    #[error("Slurm helper protocol is invalid")]
    Protocol(#[from] helper::Error),
    #[error("Slurm helper ledger metadata is invalid; do not replay")]
    Metadata,
    #[error("Slurm helper root must be initialized explicitly and only once")]
    Bootstrap,
}
impl From<serde_json::Error> for Error {
    fn from(_: serde_json::Error) -> Self {
        Self::Metadata
    }
}

/// Read one exact bounded metadata frame. The caller must provide a transport
/// reader with its own finite I/O deadline; arbitrary blocking Read cannot be
/// made deadline-safe by a library wrapper. No payload/archive is spooled here.
pub fn read_request(reader: &mut impl std::io::Read) -> Result<helper::Request> {
    let mut header = [0; 4];
    reader.read_exact(&mut header)?;
    let size = u32::from_be_bytes(header) as usize;
    if size == 0 || size > helper::MAX_METADATA_BYTES {
        return Err(helper::Error::TooLarge.into());
    }
    let mut frame = vec![0; size + 4];
    frame[..4].copy_from_slice(&header);
    reader.read_exact(&mut frame[4..])?;
    let mut extra = [0];
    if reader.read(&mut extra)? != 0 {
        return Err(helper::Error::Frame.into());
    }
    Ok(helper::decode_request(&frame)?)
}
/// Writer deadline and connection ownership remain explicit outer-shell policy.
pub fn write_reply(writer: &mut impl std::io::Write, reply: &helper::Reply) -> Result<()> {
    writer.write_all(&helper::encode_reply(reply)?)?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests;
