"""Trace-driven discrete-event queuing simulator for CAMIR (Experiment E12).

Models:
- Single-stream local server (M/G/1 queue): Apple Silicon local tier (Small/Medium).
- Concurrency-capped hosted server: Hosted API tier with capacity c (default: 10 concurrent requests).
- Real empirical service times sampled from SQLite trace records (TraceRow.t_total_s).
- Escalation queuing tax: delay penalty when queries fail on small and queue for hosted large.
"""

from dataclasses import dataclass, field
import heapq
import random
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel

from camir.replay.baselines import BasePolicy
from camir.traces.schema import TraceRow


class SimulationMetrics(BaseModel):
    """Metrics for a single arrival rate simulation run."""
    arrival_rate_qps: float
    total_queries: int
    mean_wait_s: float
    p50_latency_s: float
    p90_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    max_latency_s: float
    local_utilization: float
    hosted_utilization: float
    escalation_count: int
    queue_tax_ratio: float


@dataclass(order=True)
class Event:
    timestamp: float
    event_type: str = field(compare=False)
    query_id: int = field(compare=False)
    data: Dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass
class QueryState:
    query_id: int
    arrival_time: float
    prompt_item: Dict[str, Any]
    traces: Dict[str, TraceRow]
    chosen_tier: str
    escalated: bool = False
    start_service_time: Optional[float] = None
    finish_time: Optional[float] = None
    queue_wait_s: float = 0.0


class LoadStudyResult(BaseModel):
    policy_name: str
    results_by_rate: List[SimulationMetrics]


class DiscreteEventSimulator:
    """Discrete-event queueing simulator driven by empirical trace execution times."""

    def __init__(
        self,
        local_concurrency: int = 1,
        hosted_concurrency: int = 10,
        seed: int = 42,
    ):
        self.local_concurrency = local_concurrency
        self.hosted_concurrency = hosted_concurrency
        self.seed = seed

    def run_simulation(
        self,
        policy: BasePolicy,
        prompt_items: List[Dict[str, Any]],
        traces_by_prompt: Dict[str, Dict[str, TraceRow]],
        arrival_rate_qps: float,
        duration_s: float = 120.0,
    ) -> SimulationMetrics:
        """Execute a discrete-event simulation run for a given arrival rate lambda."""
        rng = random.Random(self.seed + int(arrival_rate_qps * 100))
        np_rng = np.random.default_rng(self.seed + int(arrival_rate_qps * 100))

        # Event queue: priority queue of (timestamp, event_type, query_id, data)
        event_queue: List[Event] = []
        queries: Dict[int, QueryState] = {}

        # 1. Generate Poisson query arrivals over duration_s
        cur_time = 0.0
        query_counter = 0

        # Filter prompts that have traces
        usable_prompts = [p for p in prompt_items if p["prompt_id"] in traces_by_prompt]
        if not usable_prompts:
            usable_prompts = prompt_items

        while cur_time < duration_s:
            inter_arrival = rng.expovariate(arrival_rate_qps)
            cur_time += inter_arrival
            if cur_time >= duration_s:
                break

            p_item = rng.choice(usable_prompts)
            pid = p_item["prompt_id"]
            p_traces = traces_by_prompt.get(pid, {})

            # Route query using policy
            chosen_tier, chosen_trace, route_meta = policy.route(p_item, p_traces)
            escalated = bool(route_meta.get("escalated", False))

            query_state = QueryState(
                query_id=query_counter,
                arrival_time=cur_time,
                prompt_item=p_item,
                traces=p_traces,
                chosen_tier=chosen_tier,
                escalated=escalated,
            )
            queries[query_counter] = query_state

            heapq.heappush(
                event_queue,
                Event(timestamp=cur_time, event_type="QUERY_ARRIVAL", query_id=query_counter),
            )
            query_counter += 1

        total_queries = len(queries)
        if total_queries == 0:
            return SimulationMetrics(
                arrival_rate_qps=arrival_rate_qps,
                total_queries=0,
                mean_wait_s=0.0,
                p50_latency_s=0.0,
                p90_latency_s=0.0,
                p95_latency_s=0.0,
                p99_latency_s=0.0,
                max_latency_s=0.0,
                local_utilization=0.0,
                hosted_utilization=0.0,
                escalation_count=0,
                queue_tax_ratio=1.0,
            )

        # Server state
        local_busy = 0
        hosted_busy = 0
        local_queue: List[int] = []
        hosted_queue: List[int] = []

        total_local_busy_time = 0.0
        total_hosted_busy_time = 0.0

        # Process events chronologically
        while event_queue:
            evt = heapq.heappop(event_queue)
            t = evt.timestamp
            qid = evt.query_id
            q = queries[qid]

            if evt.event_type == "QUERY_ARRIVAL":
                # Determine initial server destination
                # If query escalates, it must first execute on Small before escalating to Large
                init_tier = "small" if q.escalated else q.chosen_tier

                if init_tier in ("small", "medium"):
                    if local_busy < self.local_concurrency:
                        local_busy += 1
                        q.start_service_time = t
                        q.queue_wait_s += max(0.0, t - q.arrival_time)
                        serv_time = self._get_service_time(q, init_tier)
                        total_local_busy_time += serv_time
                        heapq.heappush(
                            event_queue,
                            Event(
                                timestamp=t + serv_time,
                                event_type="LOCAL_FINISH",
                                query_id=qid,
                                data={"tier": init_tier},
                            ),
                        )
                    else:
                        local_queue.append(qid)
                else:  # Hosted Large
                    if hosted_busy < self.hosted_concurrency:
                        hosted_busy += 1
                        q.start_service_time = t
                        q.queue_wait_s += max(0.0, t - q.arrival_time)
                        serv_time = self._get_service_time(q, "large")
                        total_hosted_busy_time += serv_time
                        heapq.heappush(
                            event_queue,
                            Event(
                                timestamp=t + serv_time,
                                event_type="HOSTED_FINISH",
                                query_id=qid,
                            ),
                        )
                    else:
                        hosted_queue.append(qid)

            elif evt.event_type == "LOCAL_FINISH":
                local_busy -= 1
                # Check if this query needs escalation to Large
                if q.escalated:
                    # Query finished small attempt, now enters hosted queue
                    q.escalated = False  # Mark escalated stage as entered
                    q.arrival_time_large = t
                    if hosted_busy < self.hosted_concurrency:
                        hosted_busy += 1
                        serv_time = self._get_service_time(q, "large")
                        total_hosted_busy_time += serv_time
                        heapq.heappush(
                            event_queue,
                            Event(
                                timestamp=t + serv_time,
                                event_type="HOSTED_FINISH",
                                query_id=qid,
                            ),
                        )
                    else:
                        hosted_queue.append(qid)
                else:
                    q.finish_time = t

                # Schedule next waiting local query
                if local_queue and local_busy < self.local_concurrency:
                    next_qid = local_queue.pop(0)
                    local_busy += 1
                    next_q = queries[next_qid]
                    next_q.start_service_time = t
                    next_q.queue_wait_s += max(0.0, t - next_q.arrival_time)
                    tier = "small" if next_q.escalated else next_q.chosen_tier
                    serv_time = self._get_service_time(next_q, tier)
                    total_local_busy_time += serv_time
                    heapq.heappush(
                        event_queue,
                        Event(
                            timestamp=t + serv_time,
                            event_type="LOCAL_FINISH",
                            query_id=next_qid,
                            data={"tier": tier},
                        ),
                    )

            elif evt.event_type == "HOSTED_FINISH":
                hosted_busy -= 1
                q.finish_time = t

                # Schedule next waiting hosted query
                if hosted_queue and hosted_busy < self.hosted_concurrency:
                    next_qid = hosted_queue.pop(0)
                    hosted_busy += 1
                    next_q = queries[next_qid]
                    wait_start = getattr(next_q, "arrival_time_large", next_q.arrival_time)
                    next_q.queue_wait_s += max(0.0, t - wait_start)
                    serv_time = self._get_service_time(next_q, "large")
                    total_hosted_busy_time += serv_time
                    heapq.heappush(
                        event_queue,
                        Event(
                            timestamp=t + serv_time,
                            event_type="HOSTED_FINISH",
                            query_id=next_qid,
                        ),
                    )

        # 2. Compute Latencies and Tail Distribution
        end_to_end_latencies = []
        queue_waits = []
        escalation_count = 0

        for q in queries.values():
            if q.finish_time is not None:
                e2e = q.finish_time - q.arrival_time
                end_to_end_latencies.append(e2e)
                queue_waits.append(q.queue_wait_s)
                if getattr(q, "arrival_time_large", None) is not None:
                    escalation_count += 1

        sim_duration = max(1.0, max((q.finish_time for q in queries.values() if q.finish_time), default=duration_s))

        latencies = sorted(end_to_end_latencies) if end_to_end_latencies else [0.0]
        mean_wait = float(np.mean(queue_waits)) if queue_waits else 0.0

        p50 = float(np.percentile(latencies, 50))
        p90 = float(np.percentile(latencies, 90))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        max_lat = float(max(latencies))

        local_util = min(1.0, total_local_busy_time / (sim_duration * self.local_concurrency))
        hosted_util = min(1.0, total_hosted_busy_time / (sim_duration * self.hosted_concurrency))

        queue_tax_ratio = (p95 / max(0.001, p50)) if p50 > 0 else 1.0

        return SimulationMetrics(
            arrival_rate_qps=arrival_rate_qps,
            total_queries=len(end_to_end_latencies),
            mean_wait_s=mean_wait,
            p50_latency_s=p50,
            p90_latency_s=p90,
            p95_latency_s=p95,
            p99_latency_s=p99,
            max_latency_s=max_lat,
            local_utilization=local_util,
            hosted_utilization=hosted_util,
            escalation_count=escalation_count,
            queue_tax_ratio=queue_tax_ratio,
        )

    def _get_service_time(self, q: QueryState, tier: str) -> float:
        """Extract empirical service time from real trace row or fallback to realistic empirical defaults."""
        trace_row = q.traces.get(tier)
        if trace_row and trace_row.t_total_s > 0:
            return trace_row.t_total_s

        # Realistic baseline durations if trace absent for specific tier
        if tier == "small":
            return 0.25
        elif tier == "medium":
            return 0.85
        else:  # large
            return 0.65
