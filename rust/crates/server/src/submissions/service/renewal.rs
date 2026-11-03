use super::*;
impl Service {
    pub fn renew(
        &self,
        action: &str,
        request: ml_exp_core::renewal::Request,
        context: Context<'_>,
    ) -> Result<Value> {
        request.validate(action).map_err(|_| Error::Configuration)?;
        let lock = self.lock()?;
        let original = self.load(action)?;
        let b = &original.binding;
        let latest = self
            .saved_attempt(&b.project, &b.run_id, &b.attempt_id)?
            .ok_or(Error::Authority)?;
        let name = request.record(action).map_err(|_| Error::Configuration)?;
        let saved: Option<crate::renewal::Record<Prepared>> =
            crate::renewal::read(&self.slots, &name).map_err(|_| Error::Authority)?;
        let candidate = if let Some(saved) = saved {
            if saved.original != action || saved.request != request {
                return Err(Error::Conflict);
            }
            saved.prepared
        } else {
            if latest.plan["action_id"] != action {
                return Err(Error::Conflict);
            }
            ml_exp_core::renewal::eligible(&self.actions.observe(action)?, context.now)
                .map_err(|_| Error::Conflict)?;
            let facts = self.facts_for(&b.project, &b.run_id, Some(&b.attempt_id))?;
            if facts.submitted || facts.scheduler_state != "NOT_SUBMITTED" {
                return Err(Error::Conflict);
            }
            let backend = &self.backends[&facts.backend];
            native::prepare(
                &facts.run,
                &facts.attempt,
                b.clone(),
                &backend.profile,
                &self.runtime(&b.project, &facts.backend, &facts.runtime)?,
                original.approval.clone(),
                &original.request.intent().submission_token,
                context.now,
                context.expires,
            )?
        };
        if candidate.request != original.request
            || candidate.binding != original.binding
            || candidate.approval != original.approval
        {
            return Err(Error::Conflict);
        }
        let id = candidate.plan["action_id"]
            .as_str()
            .ok_or(Error::Authority)?;
        if id == action {
            return Err(Error::Conflict);
        }
        for value in [&original, &candidate] {
            let id = value.plan["action_id"].as_str().ok_or(Error::Authority)?;
            let archive = ml_exp_core::renewal::archive(id).map_err(|_| Error::Authority)?;
            crate::renewal::retain(&self.slots, &archive, value).map_err(|_| Error::Authority)?;
        }
        crate::renewal::retain(
            &self.slots,
            &name,
            &crate::renewal::Record {
                original: action.into(),
                request: request.clone(),
                prepared: candidate.clone(),
            },
        )
        .map_err(|_| Error::Authority)?;
        self.check(&candidate)?;
        let snapshot =
            self.actions
                .save_plan(&candidate.plan, context.now, context.transition_id)?;
        self.actions.bind_slurm_request(id, &candidate.request)?;
        crate::renewal::link(&self.slots, action, id, &request).map_err(|_| Error::Authority)?;
        if latest.plan["action_id"] == action {
            lock.validate()?;
            self.slots
                .replace_regular(
                    OsStr::new(&Self::slot_name(&b.project, &b.run_id, &b.attempt_id)?),
                    &serde_json::to_vec(&candidate).map_err(|_| Error::Configuration)?,
                )
                .map_err(|_| Error::Uncertain)?;
        }
        let mut view = self.public(snapshot, context.now)?;
        view["renewed_from"] = json!(action);
        view["renewal_request_id"] = json!(request.request_id);
        view["scheduler_mutation_performed"] = json!(false);
        Ok(view)
    }
}
