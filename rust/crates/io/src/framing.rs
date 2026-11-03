//! One bounded, length-prefixed control frame over a pipe/socket. No retries,
//! background reader threads, payload spooling, or implicit timeout policy.
//!
//! The caller must exclusively own the descriptor's open file description for
//! the call: O_NONBLOCK is shared by duplicated descriptors. Original flags are
//! restored before success; errors and unwinding also attempt restoration, but
//! a failed restoration is an unavailable transport, not a restored-descriptor
//! guarantee.
//! Regular files, terminals and devices are rejected because O_NONBLOCK cannot
//! give their I/O the finite pipe deadline promised here. Filesystem/kernel
//! metadata syscalls themselves are outside a hard real-time guarantee.
#![forbid(unsafe_code)]
use rustix::{
    event::{PollFd, PollFlags, Timespec, poll},
    fs::{FileType, OFlags, fcntl_getfl, fcntl_setfl, fstat},
    io::{Errno, read, write},
};
use std::{
    io,
    os::fd::{AsFd, BorrowedFd},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

struct Nonblocking<'a> {
    fd: BorrowedFd<'a>,
    original: OFlags,
    restored: bool,
}
impl<'a> Nonblocking<'a> {
    fn new(fd: BorrowedFd<'a>) -> io::Result<Self> {
        let kind = FileType::from_raw_mode(fstat(fd)?.st_mode);
        if kind != FileType::Fifo && kind != FileType::Socket {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "frame transport must be a pipe or socket",
            ));
        }
        let original = fcntl_getfl(fd)?;
        fcntl_setfl(fd, original | OFlags::NONBLOCK)?;
        Ok(Self {
            fd,
            original,
            restored: false,
        })
    }
    fn restore(&mut self) -> io::Result<()> {
        fcntl_setfl(self.fd, self.original)?;
        self.restored = true;
        Ok(())
    }
}
impl Drop for Nonblocking<'_> {
    fn drop(&mut self) {
        if !self.restored {
            let _ = fcntl_setfl(self.fd, self.original);
        }
    }
}
fn check(deadline: Instant, cancelled: &AtomicBool) -> io::Result<()> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "frame transport cancelled",
        ));
    }
    if Instant::now() >= deadline {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "frame transport deadline exceeded",
        ));
    }
    Ok(())
}
fn wait(
    fd: BorrowedFd<'_>,
    events: PollFlags,
    deadline: Instant,
    cancelled: &AtomicBool,
) -> io::Result<()> {
    loop {
        check(deadline, cancelled)?;
        let duration = deadline
            .saturating_duration_since(Instant::now())
            .min(Duration::from_millis(20));
        let timeout = Timespec::try_from(duration)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid frame deadline"))?;
        let mut fds = [PollFd::from_borrowed_fd(fd, events)];
        match poll(&mut fds, Some(&timeout)) {
            Ok(0) => {}
            Ok(_) => {
                if fds[0].revents().contains(PollFlags::NVAL) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "frame descriptor unavailable",
                    ));
                }
                // HUP/ERR are observed by the actual read/write (EOF/EPIPE).
                return Ok(());
            }
            Err(Errno::INTR) => {}
            Err(error) => return Err(error.into()),
        }
    }
}
fn read_part(
    fd: BorrowedFd<'_>,
    bytes: &mut [u8],
    deadline: Instant,
    cancelled: &AtomicBool,
) -> io::Result<usize> {
    loop {
        check(deadline, cancelled)?;
        match read(fd, &mut *bytes) {
            Ok(count) => return Ok(count),
            Err(Errno::AGAIN) => wait(fd, PollFlags::IN, deadline, cancelled)?,
            Err(Errno::INTR) => {}
            Err(error) => return Err(error.into()),
        }
    }
}
fn read_exact(
    fd: BorrowedFd<'_>,
    mut bytes: &mut [u8],
    deadline: Instant,
    cancelled: &AtomicBool,
) -> io::Result<()> {
    while !bytes.is_empty() {
        let length = bytes.len().min(64 * 1024);
        let count = read_part(fd, &mut bytes[..length], deadline, cancelled)?;
        if count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "incomplete frame",
            ));
        }
        bytes = &mut bytes[count..];
    }
    Ok(())
}
fn bound(limit: usize) -> io::Result<()> {
    if limit == 0 || limit > u32::MAX as usize || limit.checked_add(4).is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid frame bound",
        ));
    }
    Ok(())
}
/// Return the exact encoded frame (four-byte big-endian length + body), only
/// after observing EOF. A stalled header/body/EOF shares the same deadline;
/// trailing bytes or a second frame are never silently accepted.
pub fn read_frame(
    fd: impl AsFd,
    max_payload: usize,
    deadline: Instant,
    cancelled: &AtomicBool,
) -> io::Result<Vec<u8>> {
    bound(max_payload)?;
    check(deadline, cancelled)?;
    let fd = fd.as_fd();
    let mut guard = Nonblocking::new(fd)?;
    let result = (|| {
        let mut header = [0; 4];
        read_exact(fd, &mut header, deadline, cancelled)?;
        let count = u32::from_be_bytes(header) as usize;
        if count == 0 || count > max_payload {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "frame exceeds bound",
            ));
        }
        let mut frame = Vec::new();
        frame
            .try_reserve_exact(count + 4)
            .map_err(|_| io::Error::other("frame allocation unavailable"))?;
        frame.extend_from_slice(&header);
        frame.resize(count + 4, 0);
        read_exact(fd, &mut frame[4..], deadline, cancelled)?;
        if read_part(fd, &mut [0], deadline, cancelled)? != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "trailing frame bytes",
            ));
        }
        check(deadline, cancelled)?;
        Ok(frame)
    })();
    let restored = guard.restore();
    match result {
        Ok(frame) => restored.map(|()| frame),
        Err(error) => {
            let _ = restored;
            Err(error)
        }
    }
}
/// Write one already encoded bounded frame. The per-thread SIGPIPE guard is
/// shared with process I/O; no global handler or inherited disposition is used.
/// A failed/partial write is uncertainty, never permission to replay a request.
pub fn write_frame(
    fd: impl AsFd,
    frame: &[u8],
    max_payload: usize,
    deadline: Instant,
    cancelled: &AtomicBool,
) -> io::Result<()> {
    bound(max_payload)?;
    if frame.len() < 5
        || frame.len() - 4 > max_payload
        || u32::from_be_bytes(frame[..4].try_into().expect("checked header")) as usize
            != frame.len() - 4
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid encoded frame",
        ));
    }
    check(deadline, cancelled)?;
    let fd = fd.as_fd();
    let mut guard = Nonblocking::new(fd)?;
    let result = (|| {
        let mut signals = crate::process::PipeSignals::block()?;
        let mut remaining = frame;
        while !remaining.is_empty() {
            check(deadline, cancelled)?;
            match write(fd, &remaining[..remaining.len().min(64 * 1024)]) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "frame write failed",
                    ));
                }
                Ok(count) => remaining = &remaining[count..],
                Err(Errno::AGAIN) => wait(fd, PollFlags::OUT, deadline, cancelled)?,
                Err(Errno::INTR) => {}
                Err(error) => {
                    if error == Errno::PIPE {
                        signals.mark_broken_pipe();
                    }
                    return Err(error.into());
                }
            }
        }
        check(deadline, cancelled)?;
        Ok(())
    })();
    let restored = guard.restore();
    result.and(restored)
}

/// Read just a control frame prefix when an explicitly selected protocol
/// follows it with a raw payload. Ordinary RPC must continue using read_frame.
pub fn read_prefix(
    fd: impl AsFd,
    max_payload: usize,
    deadline: Instant,
    cancelled: &AtomicBool,
) -> io::Result<Vec<u8>> {
    bound(max_payload)?;
    let fd = fd.as_fd();
    let mut guard = Nonblocking::new(fd)?;
    let result = (|| {
        let mut header = [0u8; 4];
        read_exact(fd, &mut header, deadline, cancelled)?;
        let size = u32::from_be_bytes(header) as usize;
        if size == 0 || size > max_payload {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "frame exceeds bound",
            ));
        }
        let mut frame = Vec::with_capacity(size + 4);
        frame.extend_from_slice(&header);
        frame.resize(size + 4, 0);
        read_exact(fd, &mut frame[4..], deadline, cancelled)?;
        Ok(frame)
    })();
    let restored = guard.restore();
    match result {
        Ok(v) => restored.map(|()| v),
        Err(e) => {
            let _ = restored;
            Err(e)
        }
    }
}
/// One bounded raw-payload read on an exclusively owned pipe/socket.
pub fn read_payload(
    fd: impl AsFd,
    bytes: &mut [u8],
    deadline: Instant,
    cancelled: &AtomicBool,
) -> io::Result<usize> {
    if bytes.is_empty() || bytes.len() > 65536 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid payload buffer",
        ));
    }
    let fd = fd.as_fd();
    let mut guard = Nonblocking::new(fd)?;
    let result = read_part(fd, bytes, deadline, cancelled);
    let restored = guard.restore();
    match result {
        Ok(n) => restored.map(|()| n),
        Err(e) => {
            let _ = restored;
            Err(e)
        }
    }
}
/// Copy an exact regular-file payload to an owned pipe/socket with one bounded
/// buffer and an explicit total deadline. Filesystem reads may themselves block
/// in the kernel; no unbounded application allocation or output spool is used.
pub fn copy_payload(
    mut source: impl io::Read,
    fd: impl AsFd,
    bytes: u64,
    deadline: Instant,
    cancelled: &AtomicBool,
) -> io::Result<()> {
    check(deadline, cancelled)?;
    let fd = fd.as_fd();
    let mut guard = Nonblocking::new(fd)?;
    let result = (|| {
        let mut signals = crate::process::PipeSignals::block()?;
        let mut left = bytes;
        let mut buffer = [0u8; 65536];
        while left > 0 {
            check(deadline, cancelled)?;
            let count = source.read(&mut buffer[..left.min(65536) as usize])?;
            if count == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "incomplete payload",
                ));
            }
            let mut remaining = &buffer[..count];
            while !remaining.is_empty() {
                check(deadline, cancelled)?;
                match write(fd, remaining) {
                    Ok(0) => {
                        return Err(io::Error::new(
                            io::ErrorKind::WriteZero,
                            "payload write failed",
                        ));
                    }
                    Ok(n) => remaining = &remaining[n..],
                    Err(Errno::AGAIN) => wait(fd, PollFlags::OUT, deadline, cancelled)?,
                    Err(Errno::INTR) => {}
                    Err(e) => {
                        if e == Errno::PIPE {
                            signals.mark_broken_pipe();
                        }
                        return Err(e.into());
                    }
                }
            }
            left -= count as u64;
        }
        Ok(())
    })();
    let restored = guard.restore();
    match result {
        Ok(()) => restored,
        Err(e) => {
            let _ = restored;
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests;
