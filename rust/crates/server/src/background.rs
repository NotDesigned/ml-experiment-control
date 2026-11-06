//! One cancellable observer loop, run by one thread per backend so a blocked
//! SSH call cannot delay cloud polling. No scheduler mutation, checkpoint
//! selection, automatic retry or result release path: a `Polled` only samples.
use ml_exp_core::polling::{Phase, Policy, Schedule};
use ml_exp_io::directory::FileLock;
use std::{
    collections::BTreeSet,
    io,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const REDISCOVER_SECONDS: u64 = 30;

/// How a sample should move the schedule. `busy` retries the same phase soon.
pub struct Sample {
    pub ok: bool,
    pub busy: bool,
}

/// What one backend's observer does. Errors stop at the adapter, which decides
/// what to report; the loop only needs to know how to schedule the next sample.
pub trait Polled: Send + 'static {
    /// `None` means discovery failed and the known candidates stay.
    fn discover(&self) -> Option<BTreeSet<String>>;
    fn sample(&self, action: &str, phase: Phase, stop: &AtomicBool) -> Sample;
    fn started(&self) {}
    fn stopped(&self) {}
}

/// What a thread body sees of its `Worker`.
#[derive(Clone, Default)]
pub struct Stop {
    flag: Arc<AtomicBool>,
    wake: Arc<(Mutex<()>, Condvar)>,
}
impl Stop {
    pub fn flag(&self) -> &AtomicBool {
        &self.flag
    }
    pub fn requested(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
    /// Sleep for a second unless woken. `false` means stop now.
    pub fn pause(&self) -> bool {
        let Ok(guard) = self.wake.0.lock() else {
            return false;
        };
        if self.requested() {
            return false;
        }
        let _ = self.wake.1.wait_timeout(guard, Duration::from_secs(1));
        true
    }
}

/// A named thread that is told to stop, woken and joined when this is dropped.
pub struct Worker {
    stop: Stop,
    thread: Option<JoinHandle<()>>,
}
impl Worker {
    pub fn spawn(name: &str, body: impl FnOnce(&Stop) + Send + 'static) -> io::Result<Self> {
        let stop = Stop::default();
        let inner = stop.clone();
        let thread = thread::Builder::new()
            .name(name.into())
            .spawn(move || body(&inner))?;
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.stop.flag.store(true, Ordering::Relaxed);
        if let Ok(_guard) = self.stop.wake.0.lock() {
            self.stop.wake.1.notify_all();
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Run `polled` on its own thread. Holding `leader` for the thread's lifetime is
/// what stops a second leader from starting.
pub fn observe(
    name: &str,
    polled: impl Polled,
    policy: Policy,
    leader: FileLock,
) -> io::Result<Worker> {
    Worker::spawn(name, move |stop| {
        // Moved in by name: the flock must live as long as this thread.
        let _leader = leader;
        let start = Instant::now();
        let clock = || start.elapsed().as_secs();
        let mut cycle = Cycle::new(policy);
        polled.started();
        while !stop.requested() {
            cycle.step(&polled, &clock, stop.flag());
            if !stop.pause() {
                break;
            }
        }
        polled.stopped();
    })
}

/// One pass of the loop at a time, with the clock passed in, so a test can walk
/// through minutes of schedule without waiting for them.
#[doc(hidden)]
pub struct Cycle {
    policy: Policy,
    schedule: Schedule,
    candidates: BTreeSet<String>,
    discover_at: u64,
}
impl Cycle {
    pub fn new(policy: Policy) -> Self {
        Self {
            policy,
            schedule: Schedule::default(),
            candidates: BTreeSet::new(),
            discover_at: 0,
        }
    }
    pub fn step(&mut self, polled: &impl Polled, clock: &dyn Fn() -> u64, stop: &AtomicBool) {
        let elapsed = clock();
        if elapsed >= self.discover_at {
            if let Some(found) = polled.discover() {
                self.candidates = found;
            }
            self.discover_at = elapsed.saturating_add(REDISCOVER_SECONDS);
        }
        let (next, selected) =
            std::mem::take(&mut self.schedule).select(&self.candidates, elapsed, &self.policy);
        self.schedule = next;
        if let Some((action, phase)) = selected {
            let sample = polled.sample(&action, phase, stop);
            self.schedule = std::mem::take(&mut self.schedule).outcome(
                &action,
                clock(),
                &self.policy,
                sample.ok,
                sample.busy,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ml_exp_io::directory::{Directory, LockOptions};
    use std::{cell::Cell, collections::VecDeque, ffi::OsStr, io::ErrorKind};

    #[derive(Default)]
    struct Script {
        discoveries: u32,
        answers: VecDeque<Option<BTreeSet<String>>>,
        samples: Vec<(String, Phase)>,
        busy: bool,
        hold_until_stopped: bool,
        saw_stop: bool,
        started: bool,
        stopped: bool,
    }
    #[derive(Clone, Default)]
    struct Fake(Arc<Mutex<Script>>);
    impl Fake {
        fn script(&self) -> std::sync::MutexGuard<'_, Script> {
            self.0.lock().unwrap()
        }
        fn answering(answers: Vec<Option<BTreeSet<String>>>) -> Self {
            let fake = Self::default();
            fake.script().answers = answers.into();
            fake
        }
    }
    impl Polled for Fake {
        fn discover(&self) -> Option<BTreeSet<String>> {
            let mut script = self.script();
            script.discoveries += 1;
            script.answers.pop_front().flatten()
        }
        fn sample(&self, action: &str, phase: Phase, stop: &AtomicBool) -> Sample {
            let (busy, hold) = {
                let mut script = self.script();
                script.samples.push((action.into(), phase));
                (script.busy, script.hold_until_stopped)
            };
            let deadline = Instant::now() + Duration::from_secs(30);
            while hold && !stop.load(Ordering::Relaxed) && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            self.script().saw_stop = stop.load(Ordering::Relaxed);
            Sample { ok: !busy, busy }
        }
        fn started(&self) {
            self.script().started = true;
        }
        fn stopped(&self) {
            self.script().stopped = true;
        }
    }

    fn policy() -> Policy {
        Policy {
            enabled: true,
            ..Default::default()
        }
    }
    fn one(action: &str) -> BTreeSet<String> {
        BTreeSet::from([action.to_owned()])
    }
    /// Steps once per simulated second, over `seconds`, as the thread would.
    fn walk(cycle: &mut Cycle, fake: &Fake, seconds: std::ops::Range<u64>) {
        let clock = Cell::new(0);
        for second in seconds {
            clock.set(second);
            cycle.step(fake, &|| clock.get(), &AtomicBool::new(false));
        }
    }
    fn wait_for(fake: &Fake, reached: impl Fn(&Script) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while !reached(&fake.script()) {
            assert!(
                Instant::now() < deadline,
                "the observer thread never got there"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn candidates_are_discovered_again_every_30_seconds() {
        let fake = Fake::default();
        let mut cycle = Cycle::new(policy());
        walk(&mut cycle, &fake, 0..30);
        assert_eq!(fake.script().discoveries, 1);
        walk(&mut cycle, &fake, 30..60);
        assert_eq!(fake.script().discoveries, 2);
        walk(&mut cycle, &fake, 60..61);
        assert_eq!(fake.script().discoveries, 3);
    }

    #[test]
    fn a_failed_discovery_keeps_the_old_candidates_and_a_good_one_replaces_them() {
        let fake = Fake::answering(vec![Some(one("a")), None, Some(BTreeSet::new())]);
        let mut cycle = Cycle::new(policy());
        walk(&mut cycle, &fake, 0..30);
        let before = fake.script().samples.len();
        assert!(before > 0);
        walk(&mut cycle, &fake, 30..60);
        let during = fake.script().samples.len();
        assert_eq!(fake.script().discoveries, 2);
        assert!(during > before, "None must not drop the known candidate");
        walk(&mut cycle, &fake, 60..120);
        assert_eq!(fake.script().samples.len(), during);
    }

    #[test]
    fn phases_rotate_on_a_result_and_stay_put_while_busy() {
        let fake = Fake::answering(vec![Some(one("a"))]);
        let mut cycle = Cycle::new(policy());
        fake.script().busy = true;
        walk(&mut cycle, &fake, 0..3);
        fake.script().busy = false;
        walk(&mut cycle, &fake, 3..6);
        let phases: Vec<Phase> = fake.script().samples.iter().map(|s| s.1).collect();
        assert_eq!(
            phases,
            [
                Phase::Scheduler,
                Phase::Scheduler,
                Phase::Scheduler,
                Phase::Scheduler,
                Phase::Worker,
                Phase::History
            ]
        );
    }

    #[test]
    fn a_disabled_policy_still_discovers_but_never_samples() {
        let fake = Fake::answering(vec![Some(one("a"))]);
        let mut cycle = Cycle::new(Policy::default());
        walk(&mut cycle, &fake, 0..31);
        assert_eq!(fake.script().discoveries, 2);
        assert!(fake.script().samples.is_empty());
    }

    #[test]
    fn dropping_the_worker_stops_a_held_sample_and_only_then_frees_the_leader() {
        let temp = tempfile::tempdir().unwrap();
        let directory = Directory::open(temp.path(), false).unwrap();
        let name = OsStr::new(".observer.lock");
        let lock = || {
            directory.lock(
                name,
                LockOptions {
                    create: true,
                    ..Default::default()
                },
            )
        };
        let fake = Fake::answering(vec![Some(one("a"))]);
        fake.script().hold_until_stopped = true;
        let worker = observe("test-observer", fake.clone(), policy(), lock().unwrap()).unwrap();
        wait_for(&fake, |script| !script.samples.is_empty());
        assert!(fake.script().started);
        assert!(!fake.script().stopped);
        assert_eq!(lock().unwrap_err().kind(), ErrorKind::WouldBlock);
        drop(worker);
        assert!(fake.script().saw_stop);
        assert!(fake.script().stopped);
        lock().unwrap();
    }

    /// Without the wake, each drop would wait out most of a one-second pause.
    #[test]
    fn dropping_a_worker_wakes_it_instead_of_waiting_out_its_pause() {
        let begun = Instant::now();
        for _ in 0..5 {
            let (sender, receiver) = std::sync::mpsc::channel();
            let worker = Worker::spawn("test-worker", move |stop| {
                sender.send(stop.requested()).unwrap();
                while stop.pause() {}
                sender.send(stop.requested()).unwrap();
            })
            .unwrap();
            assert!(!receiver.recv().unwrap());
            thread::sleep(Duration::from_millis(50));
            drop(worker);
            assert!(receiver.recv().unwrap());
        }
        assert!(begun.elapsed() < Duration::from_secs(3));
    }
}
