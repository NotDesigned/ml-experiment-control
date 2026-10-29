//! Narrow existing builder RPC. A Unix peer credential and socket inode are
//! checked independently; there is no shell, redirect or automatic retry.
use std::{
    io::{self, BufRead, Read, Write},
    os::{
        fd::AsRawFd,
        unix::{
            fs::{FileTypeExt, MetadataExt},
            net::UnixStream,
        },
    },
    path::Path,
    time::Duration,
};
fn invalid() -> io::Error {
    io::Error::other("private builder RPC identity or response is invalid")
}
pub fn request(
    path: &Path,
    uid: u32,
    body: &[u8],
    timeout: Duration,
) -> io::Result<(u16, Vec<u8>)> {
    let before = std::fs::symlink_metadata(path)?;
    if !path.is_absolute()
        || !before.file_type().is_socket()
        || before.uid() != uid
        || before.mode() & 0o002 != 0
        || body.is_empty()
        || body.len() > 8 * 1024 * 1024
    {
        return Err(invalid());
    }
    let mut stream = UnixStream::connect(path)?;
    let mut credentials = libc::ucred {
        pid: 0,
        uid: 0,
        gid: 0,
    };
    let mut size = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: the descriptor is live; both pointers describe initialized writable
    // storage with its exact length. SO_PEERCRED is a read-only kernel query.
    let result = unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            &mut credentials as *mut _ as *mut libc::c_void,
            &mut size,
        )
    };
    let after = std::fs::symlink_metadata(path)?;
    if result != 0
        || size as usize != std::mem::size_of::<libc::ucred>()
        || credentials.uid != uid
        || before.ino() != after.ino()
        || before.dev() != after.dev()
    {
        return Err(invalid());
    }
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    write!(
        stream,
        "POST / HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()?;
    let mut reader = io::BufReader::new(stream);
    let mut line = Vec::new();
    let mut header_bytes = 0;
    reader.by_ref().take(4097).read_until(b'\n', &mut line)?;
    if line.len() > 4096 {
        return Err(invalid());
    }
    let line = std::str::from_utf8(&line).map_err(|_| invalid())?;
    let mut fields = line.split_whitespace();
    let protocol = fields.next().ok_or_else(invalid)?;
    let status: u16 = fields
        .next()
        .ok_or_else(invalid)?
        .parse()
        .map_err(|_| invalid())?;
    if !matches!(protocol, "HTTP/1.0" | "HTTP/1.1") || !(100..600).contains(&status) {
        return Err(invalid());
    }
    let mut length = None;
    loop {
        let mut row = Vec::new();
        reader.by_ref().take(4097).read_until(b'\n', &mut row)?;
        header_bytes += row.len();
        if row.is_empty() || row.len() > 4096 || header_bytes > 16384 {
            return Err(invalid());
        }
        if row == b"\r\n" || row == b"\n" {
            break;
        }
        let row = std::str::from_utf8(&row).map_err(|_| invalid())?;
        let (key, value) = row.split_once(':').ok_or_else(invalid)?;
        if key.eq_ignore_ascii_case("content-length") {
            if length.is_some() {
                return Err(invalid());
            }
            length = Some(value.trim().parse::<usize>().map_err(|_| invalid())?);
        }
        if key.eq_ignore_ascii_case("transfer-encoding") {
            return Err(invalid());
        }
    }
    let length = length
        .filter(|n| *n <= 8 * 1024 * 1024)
        .ok_or_else(invalid)?;
    let mut bytes = vec![0u8; length];
    reader.read_exact(&mut bytes)?;
    Ok((status, bytes))
}
