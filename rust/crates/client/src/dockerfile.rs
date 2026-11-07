//! Which files the image needs. Docker sees only what COPY and ADD name, so packing just
//! those cannot miss a file the Dockerfile uses, and the image's identity then depends
//! only on them. Whatever cannot be read for certain keeps the whole directory, which is
//! always correct and only costs a rebuild when code changes.
use crate::submit::instructions;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum Split {
    /// Build from these. The whole directory is delivered to the job as its code.
    Files(Vec<String>),
    /// Build from the whole directory, and deliver nothing. Says why, for the question.
    Whole(String),
}
fn whole(why: impl Into<String>) -> Split {
    Split::Whole(why.into())
}

const INSTRUCTIONS: &[&str] = &[
    "FROM",
    "RUN",
    "CMD",
    "LABEL",
    "MAINTAINER",
    "EXPOSE",
    "ENV",
    "ARG",
    "ENTRYPOINT",
    "VOLUME",
    "USER",
    "WORKDIR",
    "STOPSIGNAL",
    "HEALTHCHECK",
    "SHELL",
];

/// The sources of one COPY or ADD, or why they cannot be known.
fn sources_of(keyword: &str, words: &str) -> Result<Vec<String>, String> {
    let mut rest = words.trim_start();
    while let Some(flagged) = rest.strip_prefix("--") {
        let (flag, after) = flagged
            .split_once(char::is_whitespace)
            .unwrap_or((flagged, ""));
        let name = flag.split('=').next().unwrap_or_default();
        match (keyword, name) {
            // Another stage or image, and nothing from this directory.
            ("COPY", "from") => return Ok(Vec::new()),
            (_, "chown" | "chmod" | "link") | ("ADD", "checksum") => {}
            _ => return Err(format!("{keyword} has the option --{name}")),
        }
        rest = after.trim_start();
    }
    let mut paths: Vec<String> = if rest.starts_with('[') {
        serde_json::from_str(rest).map_err(|_| format!("a {keyword} list cannot be read"))?
    } else {
        rest.split_whitespace().map(str::to_owned).collect()
    };
    if paths.len() < 2 {
        return Err(format!("a {keyword} cannot be read: {rest}"));
    }
    paths.pop();
    paths
        .iter()
        .map(|path| source_path(keyword, path))
        .collect()
}
fn source_path(keyword: &str, raw: &str) -> Result<String, String> {
    if raw.contains(['$', '"', '\'', '\\']) {
        return Err(format!("{keyword} {raw} needs expanding"));
    }
    if raw.contains(['*', '?', '[']) {
        return Err(format!("{keyword} {raw} is a pattern"));
    }
    if keyword == "ADD" && (raw.contains("://") || raw.starts_with("git@")) {
        return Err(format!("ADD {raw} is a download"));
    }
    if raw.starts_with('/') {
        return Err(format!("{keyword} {raw} starts at the root"));
    }
    let parts: Vec<&str> = raw
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect();
    if parts.is_empty() {
        return Err(format!("{keyword} {raw} copies the whole directory"));
    }
    if parts.contains(&"..") {
        return Err(format!("{keyword} {raw} leaves the directory"));
    }
    Ok(parts.join("/"))
}
/// What the Dockerfile takes from the directory, as it wrote it.
pub fn named_sources(dockerfile: &str) -> Result<Vec<String>, String> {
    let mut named = Vec::new();
    for line in instructions(dockerfile) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (keyword, words) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let keyword = keyword.to_ascii_uppercase();
        // A heredoc's body is not instructions, and may be in anything that runs.
        if line.contains("<<") {
            return Err(format!("{keyword} has a heredoc"));
        }
        match keyword.as_str() {
            "COPY" | "ADD" => named.extend(sources_of(&keyword, words)?),
            // A bind mount sees the directory with no COPY to say so.
            "RUN"
                if words
                    .split_whitespace()
                    .any(|word| word.starts_with("--mount")) =>
            {
                return Err("RUN --mount can read any file".into());
            }
            // Runs in a later build of this image, with some other directory.
            "ONBUILD" => return Err("ONBUILD runs in a later build".into()),
            known if INSTRUCTIONS.contains(&known) => {}
            _ => return Err(format!("a line cannot be read: {line}")),
        }
    }
    Ok(named)
}
fn dockerignores(dockerfile: &str) -> [String; 3] {
    let sibling = match dockerfile.rsplit_once('/') {
        Some((parent, _)) => format!("{parent}/.dockerignore"),
        None => ".dockerignore".into(),
    };
    [
        ".dockerignore".into(),
        sibling,
        format!("{dockerfile}.dockerignore"),
    ]
}
/// `paths` is everything packed from the directory.
pub fn split(dockerfile: &str, name: &str, paths: &[String]) -> Split {
    let named = match named_sources(dockerfile) {
        Ok(named) => named,
        Err(why) => return whole(why),
    };
    let mut chosen = BTreeSet::new();
    for source in &named {
        let inside = format!("{source}/");
        let mut matched = false;
        for path in paths {
            if *path == *source || path.starts_with(&inside) {
                chosen.insert(path.clone());
                matched = true;
            }
        }
        // A file git ignores, or one that is not there: the build must say so itself.
        if !matched {
            return whole(format!(
                "the Dockerfile names {source}, which is not uploaded"
            ));
        }
    }
    chosen.insert(name.to_owned());
    for ignore in dockerignores(name) {
        if paths.contains(&ignore) {
            chosen.insert(ignore);
        }
    }
    if chosen.len() >= paths.len() {
        return whole("the Dockerfile uses every file");
    }
    Split::Files(chosen.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    const FROM: &str = "FROM python@sha256:0000\n";
    fn paths() -> Vec<String> {
        [
            "Dockerfile",
            "pyproject.toml",
            "uv.lock",
            "src/train.py",
            "src/model/net.py",
            "configs/a.yaml",
        ]
        .map(str::to_owned)
        .to_vec()
    }
    fn of(body: &str) -> Split {
        split(&format!("{FROM}{body}"), "Dockerfile", &paths())
    }
    fn files(names: &[&str]) -> Split {
        Split::Files(names.iter().map(|name| (*name).to_owned()).collect())
    }
    #[track_caller]
    fn falls_back(body: &str, why: &str) {
        match of(body) {
            Split::Whole(found) => assert!(found.contains(why), "{body}: {found}"),
            other => panic!("{body} gave {other:?}"),
        }
    }
    #[test]
    fn only_the_named_files_and_the_dockerfile_are_kept() {
        assert_eq!(
            of("WORKDIR /w\nCOPY pyproject.toml uv.lock /w/\nRUN uv sync --no-install-project\n"),
            files(&["Dockerfile", "pyproject.toml", "uv.lock"])
        );
        assert_eq!(of("RUN true\n"), files(&["Dockerfile"]));
        // A directory is everything inside it, however it is spelled.
        for source in ["configs", "configs/", "./configs", "./configs/."] {
            assert_eq!(
                of(&format!("COPY {source} /c\n")),
                files(&["Dockerfile", "configs/a.yaml"]),
                "{source}"
            );
        }
        assert_eq!(
            of("copy --chown=1:1 --chmod=644 --link uv.lock /w\nADD uv.lock /w2\n"),
            files(&["Dockerfile", "uv.lock"])
        );
        assert_eq!(
            of("COPY [\"uv.lock\", \"pyproject.toml\", \"/w/\"]\n"),
            files(&["Dockerfile", "pyproject.toml", "uv.lock"])
        );
        assert_eq!(
            of("COPY uv.lock \\\n  # a comment\n  pyproject.toml \\\n  /w/\n"),
            files(&["Dockerfile", "pyproject.toml", "uv.lock"])
        );
    }
    #[test]
    fn another_stage_takes_nothing_from_the_directory() {
        assert_eq!(
            of(
                "COPY uv.lock /w\nFROM scratch AS two\nCOPY --from=0 /w /w\nCOPY --from=base /x /x\n"
            ),
            files(&["Dockerfile", "uv.lock"])
        );
    }
    #[test]
    fn dockerignore_variants_go_with_the_dockerfile() {
        let mut paths = paths();
        paths.extend(
            [
                "docker/train.Dockerfile",
                "docker/.dockerignore",
                ".dockerignore",
                "docker/train.Dockerfile.dockerignore",
                "other/.dockerignore",
            ]
            .map(str::to_owned),
        );
        assert_eq!(
            split(
                &format!("{FROM}COPY uv.lock /w\n"),
                "docker/train.Dockerfile",
                &paths
            ),
            files(&[
                ".dockerignore",
                "docker/.dockerignore",
                "docker/train.Dockerfile",
                "docker/train.Dockerfile.dockerignore",
                "uv.lock",
            ])
        );
    }
    #[test]
    fn whatever_cannot_be_read_for_certain_keeps_the_whole_directory() {
        for (body, why) in [
            ("COPY . /w\n", "whole directory"),
            ("COPY ./ /w\n", "whole directory"),
            ("ADD . /w\n", "whole directory"),
            ("COPY [\".\", \"/w\"]\n", "whole directory"),
            ("COPY uv.lock\n", "cannot be read"),
            ("COPY\n", "cannot be read"),
            ("COPY [\"uv.lock\", /w]\n", "list cannot be read"),
            ("COPY ../outside /w\n", "leaves the directory"),
            ("COPY src/../uv.lock /w\n", "leaves the directory"),
            ("COPY /uv.lock /w\n", "starts at the root"),
            ("COPY $NAME /w\n", "needs expanding"),
            ("COPY ${NAME}.lock /w\n", "needs expanding"),
            ("COPY \"uv.lock\" /w\n", "needs expanding"),
            ("COPY src\\train.py /w\n", "needs expanding"),
            ("COPY *.toml /w\n", "is a pattern"),
            ("COPY uv.loc? /w\n", "is a pattern"),
            ("COPY [a]v.lock /w\n", "cannot be read"),
            ("COPY s[r]c /w\n", "is a pattern"),
            ("ADD https://example.com/x.tar /w\n", "is a download"),
            ("ADD git@example.com:a/b.git /w\n", "is a download"),
            ("COPY --exclude=x uv.lock /w\n", "--exclude"),
            ("ADD --from=base /x /x\n", "--from"),
            ("RUN --mount=type=bind,target=/s make\n", "RUN --mount"),
            ("RUN <<EOF\nmake\nEOF\n", "heredoc"),
            ("COPY <<EOF /x\nbody\nEOF\n", "heredoc"),
            ("ONBUILD COPY uv.lock /w\n", "ONBUILD"),
            ("COPY missing.txt /w\n", "not uploaded"),
            ("NONSENSE here\n", "cannot be read"),
            // A blank line ends what looked like a continuation.
            ("RUN make \\\n\n  install\n", "cannot be read"),
        ] {
            falls_back(body, why);
        }
    }
    #[test]
    fn a_dockerfile_that_uses_every_file_has_nothing_to_deliver_apart() {
        let paths = ["Dockerfile", "train.py"].map(str::to_owned);
        assert!(matches!(
            split(&format!("{FROM}COPY train.py /w\n"), "Dockerfile", &paths),
            Split::Whole(_)
        ));
    }
}
