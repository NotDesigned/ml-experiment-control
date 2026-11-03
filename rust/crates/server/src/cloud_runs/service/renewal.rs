use super::*;
impl Service {
    pub fn renew(
        &self,
        action: &str,
        request: ml_exp_core::renewal::Request,
        context: crate::submissions::Context<'_>,
    ) -> cloud::Result<Value> {
        request
            .validate(action)
            .map_err(|_| cloud::Error::configuration())?;
        let lock = self.lock()?;
        let original = self.load(action)?;
        let b = &original.execution.binding;
        let latest = self
            .saved_attempt(&b.project, &b.run_id, &b.attempt_id)?
            .ok_or_else(missing)?;
        let name = request
            .record(action)
            .map_err(|_| cloud::Error::configuration())?;
        let saved: Option<crate::renewal::Record<Prepared>> =
            crate::renewal::read(&self.slots, &name).map_err(io)?;
        let candidate = if let Some(saved) = saved {
            if saved.original != action || saved.request != request {
                return Err(cloud::Error::evidence());
            }
            saved.prepared
        } else {
            if latest.plan["action_id"] != action
                || self
                    .outbox
                    .optional_status(&original.execution.job)?
                    .is_some()
            {
                return Err(cloud::Error::evidence());
            }
            ml_exp_core::renewal::eligible(&self.actions.observe(action).map_err(op)?, context.now)
                .map_err(|_| cloud::Error::evidence())?;
            let plan = action::prepare(
                &original.execution,
                &original.approval,
                context.now,
                context.expires,
            )
            .map_err(|_| cloud::Error::configuration())?;
            Prepared {
                execution: original.execution.clone(),
                approval: original.approval.clone(),
                plan,
            }
        };
        if candidate.execution != original.execution || candidate.approval != original.approval {
            return Err(cloud::Error::evidence());
        }
        self.check(&candidate)?;
        let id = candidate.plan["action_id"]
            .as_str()
            .ok_or_else(cloud::Error::evidence)?;
        if id == action {
            return Err(cloud::Error::evidence());
        }
        for value in [&original, &candidate] {
            let id = value.plan["action_id"]
                .as_str()
                .ok_or_else(cloud::Error::evidence)?;
            let archive =
                ml_exp_core::renewal::archive(id).map_err(|_| cloud::Error::evidence())?;
            crate::renewal::retain(&self.slots, &archive, value).map_err(io)?;
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
        .map_err(io)?;
        let snapshot = self
            .actions
            .save_plan(&candidate.plan, context.now, context.transition_id)
            .map_err(op)?;
        self.actions
            .bind_cloud_request(id, &candidate.execution)
            .map_err(op)?;
        crate::renewal::link(&self.slots, action, id, &request).map_err(io)?;
        if latest.plan["action_id"] == action {
            lock.validate()?;
            self.slots
                .replace_regular(
                    OsStr::new(&Self::attempt_slot(&b.project, &b.run_id, &b.attempt_id)?),
                    &encode(&candidate)?,
                )
                .map_err(io)?;
        }
        // An older repeated request never moves the pointer back over a newer renewal.
        let mut view = self.view(&candidate, snapshot, context.now)?;
        view["renewed_from"] = json!(action);
        view["renewal_request_id"] = json!(request.request_id);
        view["scheduler_mutation_performed"] = json!(false);
        Ok(view)
    }
}
