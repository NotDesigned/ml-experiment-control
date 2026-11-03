use super::CanonicalError;
use std::collections::BTreeMap;

pub(super) enum Node {
    Null,
    Bool(bool),
    Number(String),
    String(Vec<u32>),
    Array(Vec<Node>),
    Object(Vec<(Vec<u32>, Node)>),
}
struct Parser<'a> {
    text: &'a str,
    at: usize,
}
pub(super) fn parse(input: &[u8]) -> Result<Node, CanonicalError> {
    let text = std::str::from_utf8(input).map_err(|_| CanonicalError("input is not UTF-8"))?;
    let mut parser = Parser { text, at: 0 };
    let value = parser.value(0)?;
    parser.space();
    if parser.at != text.len() {
        return Err(CanonicalError("trailing JSON data"));
    }
    Ok(value)
}
impl Parser<'_> {
    fn space(&mut self) {
        while self.peek().is_some_and(|c| b" \r\n\t".contains(&c)) {
            self.at += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.at).copied()
    }
    fn take(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.at += 1;
            true
        } else {
            false
        }
    }
    fn literal(&mut self, word: &str) -> Result<(), CanonicalError> {
        if self.text[self.at..].starts_with(word) {
            self.at += word.len();
            Ok(())
        } else {
            Err(CanonicalError("invalid JSON literal"))
        }
    }
    fn value(&mut self, depth: usize) -> Result<Node, CanonicalError> {
        if depth > 512 {
            return Err(CanonicalError("JSON nesting exceeds 512"));
        }
        self.space();
        match self.peek() {
            Some(b'n') => {
                self.literal("null")?;
                Ok(Node::Null)
            }
            Some(b't') => {
                self.literal("true")?;
                Ok(Node::Bool(true))
            }
            Some(b'f') => {
                self.literal("false")?;
                Ok(Node::Bool(false))
            }
            Some(b'"') => Ok(Node::String(self.string()?)),
            Some(b'[') => {
                self.at += 1;
                self.space();
                let mut values = Vec::new();
                if self.take(b']') {
                    return Ok(Node::Array(values));
                }
                loop {
                    values.push(self.value(depth + 1)?);
                    self.space();
                    if self.take(b']') {
                        break;
                    }
                    if !self.take(b',') {
                        return Err(CanonicalError("array separator required"));
                    }
                }
                Ok(Node::Array(values))
            }
            Some(b'{') => {
                self.at += 1;
                self.space();
                let mut values: Vec<(Vec<u32>, Node)> = Vec::new();
                let mut positions: BTreeMap<Vec<u32>, usize> = BTreeMap::new();
                if self.take(b'}') {
                    return Ok(Node::Object(values));
                }
                loop {
                    self.space();
                    let key = self.string()?;
                    self.space();
                    if !self.take(b':') {
                        return Err(CanonicalError("object colon required"));
                    }
                    let value = self.value(depth + 1)?;
                    if let Some(&position) = positions.get(&key) {
                        values[position].1 = value;
                    } else {
                        positions.insert(key.clone(), values.len());
                        values.push((key, value));
                    }
                    self.space();
                    if self.take(b'}') {
                        break;
                    }
                    if !self.take(b',') {
                        return Err(CanonicalError("object separator required"));
                    }
                }
                Ok(Node::Object(values))
            }
            Some(b'-' | b'0'..=b'9') => {
                let start = self.at;
                self.take(b'-');
                if self.take(b'0') {
                } else {
                    if !self.peek().is_some_and(|c| (b'1'..=b'9').contains(&c)) {
                        return Err(CanonicalError("invalid number"));
                    }
                    while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                        self.at += 1;
                    }
                }
                if self.take(b'.') {
                    self.digits()?;
                }
                if self.take(b'e') || self.take(b'E') {
                    if !self.take(b'+') {
                        self.take(b'-');
                    }
                    self.digits()?;
                }
                Ok(Node::Number(self.text[start..self.at].into()))
            }
            _ => Err(CanonicalError("invalid JSON value")),
        }
    }
    fn digits(&mut self) -> Result<(), CanonicalError> {
        let start = self.at;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.at += 1;
        }
        if self.at == start {
            Err(CanonicalError("digits required"))
        } else {
            Ok(())
        }
    }
    fn hex(&mut self) -> Result<u32, CanonicalError> {
        let bytes = self
            .text
            .as_bytes()
            .get(self.at..self.at + 4)
            .ok_or(CanonicalError("truncated Unicode escape"))?;
        let mut code = 0;
        for &c in bytes {
            code = code * 16
                + match c {
                    b'0'..=b'9' => (c - b'0') as u32,
                    b'a'..=b'f' => (c - b'a' + 10) as u32,
                    b'A'..=b'F' => (c - b'A' + 10) as u32,
                    _ => return Err(CanonicalError("invalid Unicode escape")),
                };
        }
        self.at += 4;
        Ok(code)
    }
    fn string(&mut self) -> Result<Vec<u32>, CanonicalError> {
        if !self.take(b'"') {
            return Err(CanonicalError("string required"));
        }
        let mut codes = Vec::new();
        loop {
            match self.peek() {
                None => return Err(CanonicalError("unterminated string")),
                Some(b'"') => {
                    self.at += 1;
                    return Ok(codes);
                }
                Some(b'\\') => {
                    self.at += 1;
                    let code = match self.peek() {
                        Some(b'"') => 0x22,
                        Some(b'\\') => 0x5c,
                        Some(b'/') => 0x2f,
                        Some(b'b') => 8,
                        Some(b'f') => 12,
                        Some(b'n') => 10,
                        Some(b'r') => 13,
                        Some(b't') => 9,
                        Some(b'u') => {
                            self.at += 1;
                            let high = self.hex()?;
                            if (0xd800..=0xdbff).contains(&high)
                                && self.text[self.at..].starts_with("\\u")
                            {
                                let saved = self.at;
                                self.at += 2;
                                let low = self.hex()?;
                                if (0xdc00..=0xdfff).contains(&low) {
                                    codes.push(0x10000 + ((high - 0xd800) << 10) + (low - 0xdc00));
                                    continue;
                                }
                                self.at = saved;
                            }
                            codes.push(high);
                            continue;
                        }
                        _ => return Err(CanonicalError("invalid string escape")),
                    };
                    self.at += 1;
                    codes.push(code);
                }
                Some(c) if c < 0x20 => return Err(CanonicalError("control character in string")),
                Some(_) => {
                    let code = self.text[self.at..]
                        .chars()
                        .next()
                        .ok_or(CanonicalError("invalid string"))?;
                    codes.push(code as u32);
                    self.at += code.len_utf8();
                }
            }
        }
    }
}
