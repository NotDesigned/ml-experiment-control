//! Header-safe bearer tokens shared by client and server.
pub fn validate_bearer(token: &str, minimum: usize, maximum: usize) -> Result<(), &'static str> {
    if token.len() < minimum
        || token.len() > maximum
        || !token.bytes().all(|c| (33..=126).contains(&c))
    {
        return Err("invalid bearer token length or characters");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reject_tokens_that_cannot_be_transported_as_a_bearer_header() {
        for token in [
            "",
            "spaces are forbidden",
            "bad\0control",
            "bad\r\ninjection",
            "unicode密码",
        ] {
            assert!(validate_bearer(token, 1, 8192).is_err());
        }
        assert!(validate_bearer(&"x".repeat(32), 32, 4096).is_ok());
        assert!(validate_bearer(&"x".repeat(31), 32, 4096).is_err());
        assert!(validate_bearer(&"x".repeat(4097), 32, 4096).is_err());
    }
}
