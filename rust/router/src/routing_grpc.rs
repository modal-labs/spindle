//! Wire messages for src/spindle/proto/routing.proto. The bridge sends complete
//! snapshots, so reconnecting never reuses cache state from a previous stream.
use std::collections::HashMap;

#[derive(Clone, PartialEq, prost::Message)]
pub struct WatchRequest {}
#[derive(Clone, PartialEq, prost::Message)]
pub struct RoutingSnapshot {
    #[prost(string, tag = "1")]
    pub replica: String,
    #[prost(uint64, tag = "2")]
    pub active: u64,
    #[prost(map = "string, uint64", tag = "3")]
    pub adapters: HashMap<String, u64>,
    #[prost(message, optional, tag = "4")]
    pub work: Option<Work>,
    #[prost(double, tag = "5")]
    pub observed_at: f64,
}
#[derive(Clone, PartialEq, prost::Message)]
pub struct Work {
    #[prost(double, tag = "1")]
    pub age_seconds: f64,
    #[prost(string, repeated, tag = "2")]
    pub prefixes: Vec<String>,
    #[prost(string, repeated, tag = "3")]
    pub cpu_adapters: Vec<String>,
    #[prost(string, repeated, tag = "4")]
    pub gpu_adapters: Vec<String>,
    #[prost(string, repeated, tag = "5")]
    pub request_ids: Vec<String>,
    #[prost(string, repeated, tag = "6")]
    pub pending_adapters: Vec<String>,
    #[prost(uint64, tag = "7")]
    pub waiting_requests: u64,
    #[prost(uint64, tag = "8")]
    pub kv_used_tokens: u64,
    #[prost(uint64, tag = "9")]
    pub kv_capacity_tokens: u64,
}
impl RoutingSnapshot {
    pub fn into_replica(self, upstream: String) -> Option<crate::Replica> {
        let work = self.work?;
        if !work.age_seconds.is_finite() || work.age_seconds < 0.0 {
            return None;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_secs_f64();
        if !self.observed_at.is_finite() || self.observed_at <= 0.0 {
            return None;
        }
        Some(crate::Replica {
            replica: self.replica,
            upstream,
            active: self.active as usize,
            adapters: self
                .adapters
                .into_iter()
                .map(|(k, v)| (k, v as usize))
                .collect(),
            work: crate::Work {
                age_seconds: work.age_seconds + (now - self.observed_at).max(0.0),
                prefixes: work
                    .prefixes
                    .iter()
                    .map(|h| u128::from_str_radix(h, 16))
                    .collect::<Result<_, _>>()
                    .ok()?,
                waiting_requests: work.waiting_requests as usize,
                kv_used_tokens: work.kv_used_tokens,
                kv_capacity_tokens: work.kv_capacity_tokens,
                cpu_adapters: work.cpu_adapters.into_iter().collect(),
                gpu_adapters: work.gpu_adapters.into_iter().collect(),
                request_ids: work.request_ids.into_iter().collect(),
                pending_adapters: work.pending_adapters.into_iter().collect(),
            },
        })
    }
}
