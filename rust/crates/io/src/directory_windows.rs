//! Retained Windows directory handles deny deletion and ancestor replacement.
//! Reparse points and alternate streams are rejected at every component.
use std::{
    ffi::{OsStr, OsString},
    fs::{self, File, OpenOptions},
    io::{self, Read},
    os::windows::{
        fs::{MetadataExt, OpenOptionsExt},
        io::AsRawHandle,
    },
    path::{Component, Path, PathBuf},
    sync::Arc,
};
use windows_sys::Win32::Storage::FileSystem::*;

use crate::file_lock::acquire;
pub use crate::file_lock::{FileLock, LockOptions, Wait};

#[derive(Clone, Debug)]
pub struct Directory {
    inner: Arc<Inner>,
}
#[derive(Debug)]
struct Inner {
    path: PathBuf,
    file: File,
    _parent: Option<Directory>,
}
fn invalid() -> io::Error {
    io::Error::other("unsafe or replaced Windows directory entry")
}
fn leaf(name: &OsStr) -> io::Result<()> {
    let text = name.to_str().ok_or_else(invalid)?;
    if text.is_empty()
        || text.ends_with(['.', ' '])
        || text.contains(['/', '\\', ':', '\0'])
        || matches!(text, "." | "..")
    {
        return Err(invalid());
    }
    let stem = text.split('.').next().unwrap_or("").to_ascii_uppercase();
    if matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (stem.starts_with("COM") || stem.starts_with("LPT"))
            && stem.len() == 4
            && matches!(stem.as_bytes()[3], b'1'..=b'9')
    {
        return Err(invalid());
    }
    Ok(())
}
fn info(file: &File) -> io::Result<BY_HANDLE_FILE_INFORMATION> {
    let mut value = std::mem::MaybeUninit::uninit();
    if unsafe { GetFileInformationByHandle(file.as_raw_handle(), value.as_mut_ptr()) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { value.assume_init() })
}
pub(crate) fn same(a: &File, b: &File) -> io::Result<bool> {
    let a = info(a)?;
    let b = info(b)?;
    Ok(a.dwVolumeSerialNumber == b.dwVolumeSerialNumber
        && a.nFileIndexHigh == b.nFileIndexHigh
        && a.nFileIndexLow == b.nFileIndexLow)
}
fn open_directory(path: &Path) -> io::Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_dir() || metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(invalid());
    }
    Ok(file)
}
impl Directory {
    pub fn path(&self) -> PathBuf {
        self.inner.path.clone()
    }
    pub fn open(path: &Path, create: bool) -> io::Result<Self> {
        let path = if path.is_absolute() {
            path.to_owned()
        } else {
            std::env::current_dir()?.join(path)
        };
        let mut root = PathBuf::new();
        let mut current: Option<Self> = None;
        for part in path.components() {
            match part {
                Component::Prefix(_) | Component::RootDir => {
                    root.push(part.as_os_str());
                    if root.is_absolute() {
                        current = Some(Self {
                            inner: Arc::new(Inner {
                                path: root.clone(),
                                file: open_directory(&root)?,
                                _parent: None,
                            }),
                        });
                    }
                }
                Component::Normal(name) => {
                    current = Some(current.ok_or_else(invalid)?.child(name, create)?);
                }
                Component::CurDir => {}
                _ => return Err(invalid()),
            }
        }
        current.ok_or_else(invalid)
    }
    pub fn verify_private_parent(&self) -> io::Result<()> {
        if !same(&self.inner.file, &open_directory(&self.inner.path)?)? {
            return Err(invalid());
        }
        if let Some(parent) = &self.inner._parent {
            parent.verify_private_parent()?;
        }
        Ok(())
    }
    pub fn child(&self, name: &OsStr, create: bool) -> io::Result<Self> {
        leaf(name)?;
        self.verify_private_parent()?;
        let path = self.inner.path.join(name);
        if create && !path.try_exists()? {
            match fs::create_dir(&path) {
                Ok(()) => crate::windows_private::protect(&path)?,
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(e),
            }
        }
        Ok(Self {
            inner: Arc::new(Inner {
                file: open_directory(&path)?,
                path,
                _parent: Some(self.clone()),
            }),
        })
    }
    pub fn entry_names_limited(&self, maximum: usize) -> io::Result<Vec<OsString>> {
        self.verify_private_parent()?;
        let mut names = Vec::new();
        for entry in fs::read_dir(&self.inner.path)? {
            if names.len() >= maximum {
                return Err(invalid());
            }
            let name = entry?.file_name();
            leaf(&name)?;
            names.push(name);
        }
        names.sort();
        self.verify_private_parent()?;
        Ok(names)
    }
    pub fn open_regular_read(&self, name: &OsStr) -> io::Result<File> {
        leaf(name)?;
        self.verify_private_parent()?;
        let mut options = OpenOptions::new();
        options.read(true);
        crate::windows_private::read_options(&mut options);
        let file = options.open(self.inner.path.join(name))?;
        let metadata = file.metadata()?;
        if !metadata.is_file()
            || metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
            || info(&file)?.nNumberOfLinks != 1
        {
            return Err(invalid());
        }
        Ok(file)
    }
    pub fn validate_regular_entry(&self, name: &OsStr, file: &File) -> io::Result<()> {
        leaf(name)?;
        self.verify_private_parent()?;
        let current = OpenOptions::new()
            .access_mode(FILE_READ_ATTRIBUTES)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .open(self.inner.path.join(name))?;
        let metadata = current.metadata()?;
        if !metadata.is_file()
            || metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
            || info(&current)?.nNumberOfLinks != 1
            || !same(file, &current)?
        {
            return Err(invalid());
        }
        self.verify_private_parent()
    }
    /// Take an advisory lock on `name` through a fresh handle. Contention is
    /// returned as `WouldBlock`, which the caller maps to its own busy error.
    pub fn lock(&self, name: &OsStr, options: LockOptions<'_>) -> io::Result<FileLock> {
        leaf(name)?;
        let path = self.inner.path.join(name);
        let mut opener = OpenOptions::new();
        opener
            .read(true)
            .write(options.create || options.write)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
        let file = if options.create {
            match opener.create_new(true).open(&path) {
                Ok(file) => {
                    crate::windows_private::protect(&path)?;
                    file
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                    opener.create_new(false).open(&path)?
                }
                Err(e) => return Err(e),
            }
        } else {
            opener.open(&path)?
        };
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(invalid());
        }
        acquire(file, options.shared, options.wait)
    }
    pub fn entry_exists(&self, name: &OsStr) -> io::Result<bool> {
        leaf(name)?;
        self.verify_private_parent()?;
        match fs::symlink_metadata(self.inner.path.join(name)) {
            Ok(_) => Ok(true),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e),
        }
    }
    pub fn read_regular(&self, name: &OsStr, maximum: u64) -> io::Result<Vec<u8>> {
        let file = self.open_regular_read(name)?;
        if file.metadata()?.len() > maximum {
            return Err(invalid());
        }
        let mut bytes = Vec::new();
        file.take(maximum.saturating_add(1))
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > maximum {
            return Err(invalid());
        }
        Ok(bytes)
    }
    pub fn publish_new(&self, name: &OsStr, bytes: &[u8]) -> io::Result<()> {
        leaf(name)?;
        self.verify_private_parent()?;
        crate::write_atomic(&self.inner.path.join(name), bytes, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_handles_reject_rename_writers_and_aliases() {
        let temp = tempfile::tempdir().unwrap();
        let dir = Directory::open(temp.path(), false).unwrap();
        fs::write(temp.path().join("data"), b"training").unwrap();
        let file = dir.open_regular_read(OsStr::new("data")).unwrap();
        assert!(
            OpenOptions::new()
                .write(true)
                .open(temp.path().join("data"))
                .is_err()
        );
        assert!(fs::rename(temp.path().join("data"), temp.path().join("replaced")).is_err());
        dir.validate_regular_entry(OsStr::new("data"), &file)
            .unwrap();
        for name in ["../bad", "data:stream", "CON", "NUL.txt", "tail."] {
            assert!(dir.open_regular_read(OsStr::new(name)).is_err());
        }
    }
    #[test]
    fn independent_lock_handles_preserve_exclusion_and_leaf_identity() {
        let temp = tempfile::tempdir().unwrap();
        let dir = Directory::open(temp.path(), false).unwrap();
        let name = OsStr::new(".dispatch.lock");
        let options = LockOptions {
            create: true,
            ..Default::default()
        };
        let first = dir.lock(name, options).unwrap();
        assert_eq!(
            dir.lock(name, options).unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
        assert!(fs::rename(temp.path().join(".dispatch.lock"), temp.path().join("old")).is_err());
        dir.validate_regular_entry(name, &first).unwrap();
        drop(first);
        assert!(dir.lock(name, options).is_ok());
    }
}
