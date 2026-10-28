use super::*;
use std::{
    io::{Read, Write},
    net::Shutdown,
    os::unix::net::UnixStream,
};
fn encoded(body: &[u8]) -> Vec<u8> {
    [&(body.len() as u32).to_be_bytes()[..], body].concat()
}
fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(2)
}

#[test]
fn complete_frame_requires_eof_and_restores_original_descriptor_flags() {
    let (reader, mut writer) = UnixStream::pair().unwrap();
    let flags = fcntl_getfl(&reader).unwrap();
    let frame = encoded(b"synthetic");
    let sent = frame.clone();
    let thread = std::thread::spawn(move || {
        for byte in sent {
            writer.write_all(&[byte]).unwrap();
        }
        writer.shutdown(Shutdown::Write).unwrap();
    });
    assert_eq!(
        read_frame(&reader, 64, deadline(), &AtomicBool::new(false)).unwrap(),
        frame
    );
    assert_eq!(fcntl_getfl(&reader).unwrap(), flags);
    thread.join().unwrap();
    reader.set_nonblocking(true).unwrap();
    let flags = fcntl_getfl(&reader).unwrap();
    assert!(read_frame(&reader, 64, deadline(), &AtomicBool::new(false)).is_err());
    assert_eq!(fcntl_getfl(&reader).unwrap(), flags);
}
#[test]
fn malformed_oversized_empty_and_trailing_frames_fail_without_becoming_messages() {
    for frame in [
        vec![],
        vec![0, 0],
        vec![0, 0, 0, 0],
        vec![0, 0, 1, 0],
        vec![0, 0, 0, 2, 1],
        [encoded(b"ok"), vec![1]].concat(),
        [encoded(b"one"), encoded(b"two")].concat(),
    ] {
        let (reader, mut writer) = UnixStream::pair().unwrap();
        let flags = fcntl_getfl(&reader).unwrap();
        writer.write_all(&frame).unwrap();
        writer.shutdown(Shutdown::Write).unwrap();
        assert!(read_frame(&reader, 64, deadline(), &AtomicBool::new(false)).is_err());
        assert_eq!(fcntl_getfl(&reader).unwrap(), flags);
    }
}
#[test]
fn stalled_header_body_and_eof_share_the_deadline_and_preserve_flags() {
    for bytes in [vec![], vec![0, 0, 0, 2, 1], encoded(b"ok")] {
        let (reader, mut writer) = UnixStream::pair().unwrap();
        writer.write_all(&bytes).unwrap();
        let flags = fcntl_getfl(&reader).unwrap();
        let started = Instant::now();
        let error = read_frame(
            &reader,
            64,
            started + Duration::from_millis(30),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(fcntl_getfl(&reader).unwrap(), flags);
    }
}
#[test]
fn blocked_writer_and_cancellation_are_bounded_without_detached_io_threads() {
    let (writer, _reader) = UnixStream::pair().unwrap();
    let flags = fcntl_getfl(&writer).unwrap();
    let frame = encoded(&vec![1; 8 * 1024 * 1024]);
    let started = Instant::now();
    assert_eq!(
        write_frame(
            &writer,
            &frame,
            8 * 1024 * 1024,
            started + Duration::from_millis(30),
            &AtomicBool::new(false)
        )
        .unwrap_err()
        .kind(),
        io::ErrorKind::TimedOut
    );
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(fcntl_getfl(&writer).unwrap(), flags);
    assert_eq!(
        write_frame(
            &writer,
            &encoded(b"x"),
            64,
            deadline(),
            &AtomicBool::new(true)
        )
        .unwrap_err()
        .kind(),
        io::ErrorKind::Interrupted
    );
    let (reader, _writer) = UnixStream::pair().unwrap();
    assert_eq!(
        read_frame(&reader, 64, deadline(), &AtomicBool::new(true))
            .unwrap_err()
            .kind(),
        io::ErrorKind::Interrupted
    );
}
#[test]
fn successful_reply_is_exact_and_closed_peer_is_a_controlled_broken_pipe() {
    let (writer, mut reader) = UnixStream::pair().unwrap();
    let frame = encoded(b"reply");
    let flags = fcntl_getfl(&writer).unwrap();
    write_frame(&writer, &frame, 64, deadline(), &AtomicBool::new(false)).unwrap();
    let mut received = vec![0; frame.len()];
    reader.read_exact(&mut received).unwrap();
    assert_eq!(received, frame);
    assert_eq!(fcntl_getfl(&writer).unwrap(), flags);
    drop(reader);
    assert_eq!(
        write_frame(&writer, &frame, 64, deadline(), &AtomicBool::new(false))
            .unwrap_err()
            .kind(),
        io::ErrorKind::BrokenPipe
    );
    assert_eq!(fcntl_getfl(&writer).unwrap(), flags);
}
#[test]
fn regular_files_and_invalid_write_frames_are_rejected_before_io() {
    let file = tempfile::tempfile().unwrap();
    assert!(read_frame(&file, 64, deadline(), &AtomicBool::new(false)).is_err());
    assert!(
        write_frame(
            &file,
            &encoded(b"x"),
            64,
            deadline(),
            &AtomicBool::new(false)
        )
        .is_err()
    );
    let (writer, _reader) = UnixStream::pair().unwrap();
    for bytes in [
        vec![],
        vec![0, 0, 0, 0],
        vec![0, 0, 0, 2, 1],
        encoded(b"long"),
    ] {
        assert!(write_frame(&writer, &bytes, 3, deadline(), &AtomicBool::new(false)).is_err());
    }
    assert!(read_frame(&writer, 0, deadline(), &AtomicBool::new(false)).is_err());
}

#[test]
fn explicit_stream_prefix_preserves_raw_payload_and_rejects_stalled_streams() {
    let (reader, mut writer) = UnixStream::pair().unwrap();
    let frame = encoded(b"header");
    let body = vec![71u8; 256 * 1024];
    let want = body.clone();
    let sent = frame.clone();
    let thread = std::thread::spawn(move || {
        writer.write_all(&sent).unwrap();
        copy_payload(
            &body[..],
            &writer,
            body.len() as u64,
            deadline(),
            &AtomicBool::new(false),
        )
        .unwrap();
        writer.shutdown(Shutdown::Write).unwrap();
    });
    assert_eq!(
        read_prefix(&reader, 64, deadline(), &AtomicBool::new(false)).unwrap(),
        frame
    );
    let mut actual = Vec::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = read_payload(&reader, &mut buffer, deadline(), &AtomicBool::new(false)).unwrap();
        if n == 0 {
            break;
        }
        actual.extend_from_slice(&buffer[..n]);
    }
    assert_eq!(actual, want);
    thread.join().unwrap();
    let (reader, _writer) = UnixStream::pair().unwrap();
    assert_eq!(
        read_payload(
            &reader,
            &mut [0],
            Instant::now() + Duration::from_millis(20),
            &AtomicBool::new(false)
        )
        .unwrap_err()
        .kind(),
        io::ErrorKind::TimedOut
    );
}
