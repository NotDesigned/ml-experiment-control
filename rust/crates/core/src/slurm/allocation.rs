use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{Error, component, printable};

pub const ALLOCATION_QUEUE_MARKER: &str = "\nML_EXPD_QUEUE\n";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceCount {
    pub total: u64,
    pub allocated: u64,
    pub unallocated: u64,
}
impl ResourceCount {
    fn new(total: u64, allocated: u64) -> Result<Self, Error> {
        let unallocated = total
            .checked_sub(allocated)
            .ok_or(Error::Evidence("allocation exceeds total"))?;
        Ok(Self {
            total,
            allocated,
            unallocated,
        })
    }
    fn add(&mut self, other: &Self) -> Result<(), Error> {
        self.total = self
            .total
            .checked_add(other.total)
            .ok_or(Error::Evidence("allocation overflow"))?;
        self.allocated = self
            .allocated
            .checked_add(other.allocated)
            .ok_or(Error::Evidence("allocation overflow"))?;
        self.unallocated = self
            .unallocated
            .checked_add(other.unallocated)
            .ok_or(Error::Evidence("allocation overflow"))?;
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationResources {
    pub gpus: ResourceCount,
    pub cpus: ResourceCount,
    pub memory_bytes: ResourceCount,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationNode {
    pub node: String,
    pub state: String,
    pub resources: AllocationResources,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllocationSnapshot {
    pub resources: AllocationResources,
    pub nodes: Vec<AllocationNode>,
    /// Counts visible squeue rows, including a compressed array as one row.
    /// It is not a count of individual array tasks or an admission estimate.
    pub pending_queue_entries: usize,
}
fn count(value: &str) -> Result<u64, Error> {
    if value.is_empty() || !value.bytes().all(|c| c.is_ascii_digit()) {
        return Err(Error::Evidence("scheduler count"));
    }
    value
        .parse()
        .map_err(|_| Error::Evidence("scheduler count overflow"))
}
fn required<'a>(fields: &'a BTreeMap<&str, &str>, name: &str) -> Result<&'a str, Error> {
    fields
        .get(name)
        .copied()
        .ok_or(Error::Evidence("allocation field missing"))
}
fn tres(value: &str) -> Result<BTreeMap<&str, &str>, Error> {
    let mut result = BTreeMap::new();
    for part in value.split(',').filter(|part| !part.is_empty()) {
        let Some((key, value)) = part.split_once('=') else {
            return Err(Error::Evidence("TRES field"));
        };
        if key.is_empty() || value.is_empty() || result.insert(key, value).is_some() {
            return Err(Error::Evidence("duplicate or empty TRES"));
        }
    }
    Ok(result)
}
/// Parse the deployed `scontrol show nodes --oneliner` + queue marker protocol.
/// Down/drained nodes remain part of scheduler totals. Unallocated is arithmetic
/// headroom and never a promise of immediately usable GPUs.
pub fn parse_allocation(output: &str, partition: &str) -> Result<AllocationSnapshot, Error> {
    if !component(partition) {
        return Err(Error::Configuration("partition"));
    }
    if output.len() > 1024 * 1024 {
        return Err(Error::Evidence("allocation size"));
    }
    let Some((nodes_text, queue)) = output.split_once(ALLOCATION_QUEUE_MARKER) else {
        return Err(Error::Evidence("queue marker missing"));
    };
    if queue.contains(ALLOCATION_QUEUE_MARKER) {
        return Err(Error::Evidence("duplicate queue marker"));
    }
    let mut nodes = Vec::new();
    let mut names = BTreeSet::new();
    for line in nodes_text.lines() {
        let mut values = BTreeMap::new();
        for item in line.split_whitespace() {
            if let Some((key, value)) = item.split_once('=') {
                if values.insert(key, value).is_some() {
                    return Err(Error::Evidence("duplicate allocation field"));
                }
            }
        }
        if !values
            .get("Partitions")
            .is_some_and(|v| v.split(',').any(|p| p == partition))
        {
            continue;
        }
        let name = required(&values, "NodeName")?;
        if !component(name) || !names.insert(name.to_owned()) {
            return Err(Error::Evidence("duplicate or invalid node"));
        }
        let configured = tres(required(&values, "CfgTRES")?)?;
        let allocated = tres(required(&values, "AllocTRES")?)?;
        let cpu_total = count(required(&values, "CPUTot")?)?;
        let cpu_allocated = count(required(&values, "CPUAlloc")?)?;
        let memory_total = count(required(&values, "RealMemory")?)?
            .checked_mul(1024 * 1024)
            .ok_or(Error::Evidence("memory overflow"))?;
        let memory_allocated = count(required(&values, "AllocMem")?)?
            .checked_mul(1024 * 1024)
            .ok_or(Error::Evidence("memory overflow"))?;
        if allocated.is_empty() && (cpu_allocated != 0 || memory_allocated != 0) {
            return Err(Error::Evidence("allocation evidence missing"));
        }
        let gpu_total = count(required(&configured, "gres/gpu")?)?;
        let gpu_allocated = count(allocated.get("gres/gpu").unwrap_or(&"0"))?;
        let state = required(&values, "State")?;
        if !printable(state, 128) {
            return Err(Error::Evidence("node state"));
        }
        nodes.push(AllocationNode {
            node: name.into(),
            state: state.into(),
            resources: AllocationResources {
                gpus: ResourceCount::new(gpu_total, gpu_allocated)?,
                cpus: ResourceCount::new(cpu_total, cpu_allocated)?,
                memory_bytes: ResourceCount::new(memory_total, memory_allocated)?,
            },
        });
    }
    if nodes.is_empty() {
        return Err(Error::Evidence("partition not visible"));
    }
    let mut resources = AllocationResources {
        gpus: ResourceCount::new(0, 0)?,
        cpus: ResourceCount::new(0, 0)?,
        memory_bytes: ResourceCount::new(0, 0)?,
    };
    for node in &nodes {
        resources.gpus.add(&node.resources.gpus)?;
        resources.cpus.add(&node.resources.cpus)?;
        resources.memory_bytes.add(&node.resources.memory_bytes)?;
    }
    let mut pending_queue_entries = 0;
    for row in queue.lines().filter(|line| !line.trim().is_empty()) {
        // Validate the limited ID/compressed-array output without exposing IDs.
        if !row.as_bytes().first().is_some_and(u8::is_ascii_digit)
            || row.len() > 256
            || !row
                .bytes()
                .all(|c| c.is_ascii_digit() || b"_[],-%:".contains(&c))
        {
            return Err(Error::Evidence("pending queue row"));
        }
        pending_queue_entries += 1;
    }
    Ok(AllocationSnapshot {
        resources,
        nodes,
        pending_queue_entries,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SnapshotState {
    Observed,
    Stale,
    Unknown,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AllocationObservation {
    pub status: SnapshotState,
    pub data: Option<AllocationSnapshot>,
    pub observed_at: Option<String>,
    pub checked_at: Option<String>,
    pub error: Option<String>,
    pub admission_guaranteed: bool,
}
/// Pure cache-result reduction. The owner separately handles TTL/locking and
/// supplies time as data. Failure retains successful bytes and their original
/// timestamp; a read or failed query does not manufacture a fresh observation.
pub fn allocation_observation(
    previous: Option<&AllocationObservation>,
    current: Result<AllocationSnapshot, Error>,
    checked_at: Option<&str>,
) -> AllocationObservation {
    match current {
        Ok(data) => AllocationObservation {
            status: SnapshotState::Observed,
            data: Some(data),
            observed_at: checked_at.map(str::to_owned),
            checked_at: checked_at.map(str::to_owned),
            error: None,
            admission_guaranteed: false,
        },
        Err(_) => {
            let retained = previous.filter(|p| p.data.is_some());
            AllocationObservation {
                status: if retained.is_some() {
                    SnapshotState::Stale
                } else {
                    SnapshotState::Unknown
                },
                data: retained.and_then(|p| p.data.clone()),
                observed_at: retained.and_then(|p| p.observed_at.clone()),
                checked_at: checked_at.map(str::to_owned),
                error: Some("SCHEDULER_QUERY_FAILED".into()),
                admission_guaranteed: false,
            }
        }
    }
}
