//! Dataset delivery: upload channels, the READY callback, and publication, which on
//! PAI materializes the uploaded archive through the collector. Copy intents survive
//! response loss; GET never creates a resource.
use super::*;
use ml_exp_core::datasets::{Receipt, import::DataChannel};
impl Service {
    pub(super) fn data_channel(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
    ) -> cloud::Result<DataChannel> {
        self.exchange
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?
            .dataset_channel(&self.workspace, project, executor, asset, "download")
    }
    fn data_auth(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
        token: &str,
        access: &str,
    ) -> cloud::Result<datasets::Saved> {
        use subtle::ConstantTimeEq;
        let expected = self
            .exchange
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?
            .dataset_channel(&self.workspace, project, executor, asset, access)?;
        if !bool::from(expected.token.as_bytes().ct_eq(token.as_bytes())) {
            return Err(cloud::Error {
                code: "DATASET_AUTH_REQUIRED".into(),
                http_status: Some(401),
                uncertain: false,
            });
        }
        self.data_saved(project, executor, asset)?
            .filter(|s| s.archive_verified)
            .ok_or_else(missing)
    }
    pub fn data_download_ticket(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
        token: &str,
        part: u64,
    ) -> cloud::Result<Value> {
        let saved = self.data_auth(project, executor, asset, token, "download")?;
        self.objects
            .as_ref()
            .ok_or_else(cloud::Error::configuration)?
            .data_ticket(project, executor, &saved.state.definition, part)
            .map_err(io)
    }
    pub fn data_ready(
        &self,
        project: &str,
        executor: &str,
        asset: &str,
        token: &str,
        receipt: Receipt,
    ) -> cloud::Result<Value> {
        let _lock = self.lock("data_delivery.data_ready")?;
        let mut saved = self.data_auth(project, executor, asset, token, "ready")?;
        let definition = &self
            .backends
            .get(executor)
            .ok_or_else(cloud::Error::configuration)?
            .definition;
        receipt
            .bind(project, &definition.target()?, &definition.data_root()?)
            .map_err(why!(cloud::Error::evidence()))?;
        if saved.copy_request_id.is_none()
            || saved.copy_runtime.is_none() && (!saved.image_started || saved.image.is_none())
        {
            return Err(cloud::Error::evidence());
        }
        saved.state = saved
            .state
            .publish(receipt)
            .map_err(why!(cloud::Error::evidence()))?;
        saved.diagnostic = None;
        self.slots
            .replace_regular(
                OsStr::new(&Self::data_name(project, executor, asset)?),
                &encode(&saved)?,
            )
            .map_err(io)?;
        Ok(json!({"asset_id":asset,"status":"READY"}))
    }
    pub fn publish_dataset(
        self: &Arc<Self>,
        project: &str,
        executor: &str,
        asset: &str,
        request_id: &str,
        runtime_id: Option<&str>,
    ) -> cloud::Result<Value> {
        let _effect = self
            .slots
            .lock(
                OsStr::new(&format!(
                    "{}.delivery.lock",
                    Self::data_name(project, executor, asset)?
                )),
                LockOptions {
                    create: true,
                    ..Default::default()
                },
            )
            .map_err(why_lock!(cloud::Error::evidence()))?;
        ml_exp_core::submission::validate_submission_token(request_id)
            .map_err(why!(cloud::Error::configuration()))?;
        if !self.enabled {
            return Err(cloud::Error::configuration());
        }
        self.backends
            .get(executor)
            .ok_or_else(cloud::Error::configuration)?;
        let saved = self
            .data_saved(project, executor, asset)?
            .filter(|s| s.archive_verified)
            .ok_or_else(missing)?;
        if saved.state.ready.is_some() {
            return Ok(Self::data_view(saved));
        }
        self.materialize_pai(project, executor, asset, request_id, runtime_id)
    }
}
