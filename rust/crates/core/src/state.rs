use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RunState {
    NotSubmitted,
    Created,
    Submitting,
    Queued,
    Starting,
    Running,
    Evaluating,
    Succeeded,
    Failed,
    Preempted,
    Cancelled,
    Unknown,
}

impl RunState {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Preempted | Self::Cancelled
        )
    }

    /// Match the original Attempt state machine, including sampled skips and requeue.
    /// A retry is a *new* Attempt, never a transition away from a terminal state.
    pub fn allows(self, next: Self) -> bool {
        if self.is_terminal() {
            return self == next;
        }
        match self {
            Self::NotSubmitted => next == Self::Created,
            Self::Created | Self::Unknown => next != Self::NotSubmitted,
            Self::Submitting => !matches!(next, Self::NotSubmitted | Self::Created),
            Self::Queued | Self::Starting => {
                !matches!(next, Self::NotSubmitted | Self::Created | Self::Submitting)
            }
            Self::Running => !matches!(
                next,
                Self::NotSubmitted | Self::Created | Self::Submitting | Self::Starting
            ),
            Self::Evaluating => {
                next.is_terminal() || matches!(next, Self::Evaluating | Self::Unknown)
            }
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid Attempt transition {from:?} -> {to:?}")]
pub struct InvalidTransition {
    pub from: RunState,
    pub to: RunState,
}

/// Pure reducer: callers decide how and whether to persist its returned value.
/// Invalid input leaves the caller's original state untouched; no automatic retry.
pub fn transition(current: RunState, observed: RunState) -> Result<RunState, InvalidTransition> {
    if current.allows(observed) {
        Ok(observed)
    } else {
        Err(InvalidTransition {
            from: current,
            to: observed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reducer_is_deterministic_and_never_reopens_a_terminal_attempt() {
        let original = RunState::Succeeded;
        assert_eq!(
            transition(original, RunState::Running),
            Err(InvalidTransition {
                from: original,
                to: RunState::Running
            })
        );
        assert_eq!(original, RunState::Succeeded);
        assert_eq!(
            transition(RunState::Running, RunState::Queued),
            Ok(RunState::Queued)
        );
        assert_eq!(
            transition(RunState::Running, RunState::Queued),
            transition(RunState::Running, RunState::Queued)
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    None,
    Transport,
    Scheduler,
    Preemption,
    Resource,
    Configuration,
    Model,
    Evaluation,
    Unknown,
}
