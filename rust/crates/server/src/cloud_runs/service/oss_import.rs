//! Existing approved OSS objects enter the same immutable dataset path as
//! uploads. Inspect first, transfer with one fixed-size buffer, then verify.
use super::*;
use ml_exp_core::datasets::{Command, oss::Import};
use sha2::{Digest, Sha256};
use std::io::Read;

impl Service {
    pub fn import_oss_dataset(
        &self,
        project: &str,
        executor: &str,
        request: Import,
    ) -> cloud::Result<Value> {
        self.project_hash(project)?;
        if !self.backends.contains_key(executor) || !self.has_datasets() {
            return Err(cloud::Error::configuration());
        }
        let objects = self
            .objects
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?;
        objects
            .validate_import(&request)
            .map_err(|_| cloud::Error {
                code: "OSS_DATASET_SCOPE_DENIED".into(),
                http_status: Some(422),
                uncertain: false,
            })?;
        let asset = request.asset_id();
        let definition = if let Some(saved) = self.data_saved(project, executor, &asset)? {
            request
                .matches(&saved.state.definition)
                .map_err(|_| cloud::Error::evidence())?;
            if saved
                .import_source
                .as_ref()
                .is_some_and(|old| old != &request)
            {
                return Err(cloud::Error::evidence());
            }
            if saved.archive_verified {
                return Ok(Self::data_view(saved));
            }
            saved.state.definition
        } else {
            let definition = objects.inspect_import(&request).map_err(|error| {
                let verification = error.kind() == std::io::ErrorKind::InvalidData;
                cloud::Error {
                    code: if verification {
                        "OSS_DATASET_VERIFICATION_FAILED"
                    } else {
                        "OSS_DATASET_READ_FAILED"
                    }
                    .into(),
                    http_status: Some(if verification { 422 } else { 503 }),
                    uncertain: false,
                }
            })?;
            self.dataset_transfer(
                project,
                executor,
                Command::Begin {
                    definition: definition.clone(),
                },
                &[],
            )?;
            definition
        };
        {
            let lock = self.lock()?;
            let mut saved = self
                .data_saved(project, executor, &asset)?
                .ok_or_else(missing)?;
            if saved.state.definition != definition
                || saved
                    .import_source
                    .as_ref()
                    .is_some_and(|old| old != &request)
            {
                return Err(cloud::Error::evidence());
            }
            saved.import_source = Some(request.clone());
            lock.validate()?;
            self.slots
                .replace_regular(
                    OsStr::new(&Self::data_name(project, executor, &asset)?),
                    &encode(&saved)?,
                )
                .map_err(io)?;
        }
        let mut source = objects.open_import(&request).map_err(io)?;
        let mut whole = Sha256::new();
        for number in 0..definition
            .part_count()
            .map_err(|_| cloud::Error::evidence())?
        {
            let size = definition
                .part_size(number)
                .map_err(|_| cloud::Error::evidence())?;
            let mut payload = vec![0; size as usize];
            source.read_exact(&mut payload).map_err(io)?;
            whole.update(&payload);
            self.dataset_transfer(
                project,
                executor,
                Command::Part {
                    asset_id: asset.clone(),
                    number,
                    sha256: format!("{:x}", Sha256::digest(&payload)),
                    bytes: size,
                },
                &payload,
            )?;
        }
        if source.read(&mut [0]).map_err(io)? != 0
            || format!("{:x}", whole.finalize()) != request.sha256
        {
            return Err(cloud::Error::evidence());
        }
        self.dataset_transfer(
            project,
            executor,
            Command::Complete { asset_id: asset },
            &[],
        )
    }
}
