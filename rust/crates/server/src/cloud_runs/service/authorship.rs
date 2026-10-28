use super::*;
impl Service {
    pub fn create_run(
        &self,
        project: &str,
        request: run::Create,
        now: &str,
    ) -> cloud::Result<Value> {
        if !self.creation {
            return Err(cloud::Error {
                code: "CLOUD_RUN_CREATION_DISABLED".into(),
                http_status: Some(403),
                uncertain: false,
            });
        }
        let lock = self.lock()?;
        let root = self.authority(project, &request.run_id)?;
        let exists = root.entry_exists(OsStr::new(&request.run_id)).map_err(io)?;
        let route = if exists {
            read(
                &root.child(OsStr::new(&request.run_id), false).map_err(io)?,
                "manifest.json",
            )?["tracking_route"]
                .clone()
        } else {
            ml_exp_core::tracking::resolve(request.wandb.as_ref(), &self.tracking, project)
        };
        let docs = self.documents(project, &request, &route, now)?;
        let checked = |directory: &Directory, reused: bool| -> cloud::Result<Value> {
            let run = read(directory, "manifest.json")?;
            let original = read(directory, "request.json")?;
            let attempt = read(
                &directory
                    .child(OsStr::new("attempts"), false)
                    .map_err(io)?
                    .child(OsStr::new("attempt-001"), false)
                    .map_err(io)?,
                "attempt.json",
            )?;
            if !creation::same_creation(&docs, &run, &attempt, &original) {
                return Err(cloud::Error::evidence());
            }
            creation::receipt(project, &request, &run, &attempt, reused)
                .map_err(|_| cloud::Error::evidence())
        };
        if exists {
            return checked(
                &root.child(OsStr::new(&request.run_id), false).map_err(io)?,
                true,
            );
        }
        let (_, random) =
            projects::transition_context().map_err(|_| cloud::Error::configuration())?;
        let stage = self
            .slots
            .create_new_child(OsStr::new(&format!("run-{random}")))
            .map_err(io)?;
        write(&stage, "manifest.json", &docs.run)?;
        write(&stage, "request.json", &docs.request)?;
        write(&stage, "status.json", &docs.status)?;
        let attempts = stage.create_new_child(OsStr::new("attempts")).map_err(io)?;
        let attempt = attempts
            .create_new_child(OsStr::new("attempt-001"))
            .map_err(io)?;
        write(&attempt, "attempt.json", &docs.attempt)?;
        write(&attempt, "status.json", &docs.status)?;
        self.authority(project, &request.run_id)?;
        lock.validate()?;
        let current = self.documents(project, &request, &route, now)?;
        if current.run != docs.run || current.attempt != docs.attempt {
            return Err(cloud::Error::evidence());
        }
        let published = stage
            .publish_directory(&root, OsStr::new(&request.run_id))
            .map_err(|_| cloud::Error {
                code: "CLOUD_RUN_CREATION_UNCERTAIN".into(),
                http_status: None,
                uncertain: true,
            })?;
        checked(&published, false)
    }
}
