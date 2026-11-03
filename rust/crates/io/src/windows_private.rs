//! Windows file boundaries: reject reparse points and protect credential DACLs.
use std::{
    fs::{File, OpenOptions},
    io,
    os::windows::{
        fs::{MetadataExt, OpenOptionsExt},
        io::AsRawHandle,
    },
    path::Path,
};
use windows_sys::Win32::System::SystemServices::{ACCESS_ALLOWED_ACE_TYPE, ACCESS_DENIED_ACE_TYPE};
use windows_sys::Win32::{
    Foundation::{CloseHandle, HANDLE, LocalFree},
    Security::{self, Authorization::*},
    Storage::FileSystem::*,
    System::Threading::{GetCurrentProcess, OpenProcessToken},
};

pub(crate) fn read_options(options: &mut OpenOptions) {
    // A retained read handle forbids writers, rename and deletion during use.
    options
        .share_mode(FILE_SHARE_READ)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
}
fn invalid() -> io::Error {
    io::Error::other("private Windows file ACL or reparse identity is invalid")
}
struct Allocation(*mut core::ffi::c_void);
impl Drop for Allocation {
    fn drop(&mut self) {
        unsafe {
            LocalFree(self.0);
        }
    }
}
struct Token(HANDLE);
impl Drop for Token {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}
fn user() -> io::Result<Vec<usize>> {
    let mut token = std::ptr::null_mut();
    // SAFETY: all output pointers refer to initialized writable storage.
    if unsafe { OpenProcessToken(GetCurrentProcess(), Security::TOKEN_QUERY, &mut token) } == 0 {
        return Err(io::Error::last_os_error());
    }
    let token = Token(token);
    let mut size = 0;
    unsafe {
        Security::GetTokenInformation(
            token.0,
            Security::TokenUser,
            std::ptr::null_mut(),
            0,
            &mut size,
        );
    }
    if size == 0 || size > 65536 {
        return Err(invalid());
    }
    let mut value = vec![0usize; (size as usize).div_ceil(std::mem::size_of::<usize>())];
    if unsafe {
        Security::GetTokenInformation(
            token.0,
            Security::TokenUser,
            value.as_mut_ptr().cast(),
            size,
            &mut size,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(value)
}
fn sid(value: &[usize]) -> Security::PSID {
    unsafe { (*(value.as_ptr().cast::<Security::TOKEN_USER>())).User.Sid }
}
pub(crate) fn validate(file: &File, require_owner: bool) -> io::Result<()> {
    if !file.metadata()?.is_file()
        || file.metadata()?.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    {
        return Err(invalid());
    }
    let current = user()?;
    let mut owner = std::ptr::null_mut();
    let mut acl = std::ptr::null_mut();
    let mut descriptor = std::ptr::null_mut();
    let status = unsafe {
        GetSecurityInfo(
            file.as_raw_handle(),
            SE_FILE_OBJECT,
            Security::OWNER_SECURITY_INFORMATION | Security::DACL_SECURITY_INFORMATION,
            &mut owner,
            std::ptr::null_mut(),
            &mut acl,
            std::ptr::null_mut(),
            &mut descriptor,
        )
    };
    if status != 0 {
        return Err(io::Error::from_raw_os_error(status as i32));
    }
    let _descriptor = Allocation(descriptor);
    if acl.is_null() || require_owner && unsafe { Security::EqualSid(owner, sid(&current)) } == 0 {
        return Err(invalid());
    }
    for index in 0..unsafe { (*acl).AceCount } as u32 {
        let mut ace = std::ptr::null_mut();
        if unsafe { Security::GetAce(acl, index, &mut ace) } == 0 {
            return Err(invalid());
        }
        let header = unsafe { &*ace.cast::<Security::ACE_HEADER>() };
        if header.AceType == ACCESS_DENIED_ACE_TYPE as u8 {
            continue;
        }
        if header.AceType != ACCESS_ALLOWED_ACE_TYPE as u8 {
            return Err(invalid());
        }
        let allowed = unsafe { &*ace.cast::<Security::ACCESS_ALLOWED_ACE>() };
        let principal = (&allowed.SidStart as *const u32).cast_mut().cast();
        let trusted = unsafe {
            Security::EqualSid(principal, sid(&current)) != 0
                || Security::IsWellKnownSid(principal, Security::WinLocalSystemSid) != 0
                || Security::IsWellKnownSid(principal, Security::WinBuiltinAdministratorsSid) != 0
        };
        if !trusted && allowed.Mask != 0 {
            return Err(invalid());
        }
    }
    Ok(())
}
pub(crate) fn protect(path: &Path) -> io::Result<()> {
    let current = user()?;
    let mut text = std::ptr::null_mut();
    if unsafe { ConvertSidToStringSidW(sid(&current), &mut text) } == 0 {
        return Err(io::Error::last_os_error());
    }
    let text_allocation = Allocation(text.cast());
    let mut length = 0;
    while unsafe { *text.add(length) } != 0 {
        length += 1;
        if length > 512 {
            return Err(invalid());
        }
    }
    let user = String::from_utf16(unsafe { std::slice::from_raw_parts(text, length) })
        .map_err(|_| invalid())?;
    drop(text_allocation);
    let text = format!("D:P(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;{user})")
        .encode_utf16()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let mut descriptor = std::ptr::null_mut();
    if unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            text.as_ptr(),
            SDDL_REVISION_1,
            &mut descriptor,
            std::ptr::null_mut(),
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    let _allocation = Allocation(descriptor);
    let mut present = 0;
    let mut defaulted = 0;
    let mut acl = std::ptr::null_mut();
    if unsafe {
        Security::GetSecurityDescriptorDacl(descriptor, &mut present, &mut acl, &mut defaulted)
    } == 0
        || present == 0
        || acl.is_null()
    {
        return Err(invalid());
    }
    let file = OpenOptions::new()
        // WRITE_DAC, WRITE_OWNER, READ_CONTROL and FILE_READ_ATTRIBUTES.
        .access_mode(0x0004_0000 | 0x0008_0000 | 0x0002_0000 | FILE_READ_ATTRIBUTES)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)?;
    if file.metadata()?.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(invalid());
    }
    let status = unsafe {
        SetSecurityInfo(
            file.as_raw_handle(),
            SE_FILE_OBJECT,
            Security::OWNER_SECURITY_INFORMATION
                | Security::DACL_SECURITY_INFORMATION
                | Security::PROTECTED_DACL_SECURITY_INFORMATION,
            sid(&current),
            std::ptr::null_mut(),
            acl,
            std::ptr::null_mut(),
        )
    };
    if status != 0 {
        return Err(io::Error::from_raw_os_error(status as i32));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn private_token_rejects_a_foreign_reader() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("token");
        crate::write_atomic(&path, b"synthetic-test-token", true).unwrap();
        assert_eq!(
            crate::read_private(&path, 128, true).unwrap(),
            b"synthetic-test-token"
        );
        let result = std::process::Command::new("icacls")
            .arg(&path)
            .args(["/grant", "*S-1-1-0:(R)"])
            .output()
            .unwrap();
        assert!(result.status.success());
        assert!(crate::read_private(&path, 128, true).is_err());
    }
}
