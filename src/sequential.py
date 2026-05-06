"""
Sequential SimPy simulation with dynamic congestion-based rerouting
and traffic light cycles at every intersection.

Each intersection has:
  - A SimPy Resource (queuing / capacity limit)
  - A traffic light with GREEN and RED phases (fixed offset per node so
    not all signals switch simultaneously – realistic staggered timing)

Vehicles recalculate their path every REROUTE_INTERVAL hops using current
BPR-adjusted edge weights.  This creates O(V log V) load per vehicle per
reroute, giving a realistically heavy workload for the benchmark.
"""

import time
from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import networkx as nx
import numpy as np
import simpy

import config
from src.network import CityNetwork

REROUTE_INTERVAL = config.REROUTE_INTERVAL
_PERIOD = config.TRAFFIC_LIGHT_GREEN + config.TRAFFIC_LIGHT_RED


# ---------------------------------------------------------------------------
def _spawn_time(rng: np.random.Generator, idx: int, total: int) -> float:
    if idx < total * config.RUSH_HOUR_FRACTION:
        t = rng.normal(
            (config.RUSH_HOUR_START + config.RUSH_HOUR_END) / 2,
            (config.RUSH_HOUR_END - config.RUSH_HOUR_START) / 6,
        )
        return float(np.clip(t, 0, config.SIM_DURATION))
    return float(rng.uniform(0, config.SIM_DURATION))


def _random_node(rng: np.random.Generator, nodes: List[Tuple]) -> Tuple:
    return nodes[rng.integers(len(nodes))]


# ---------------------------------------------------------------------------
def _dynamic_shortest_path(
    graph: nx.DiGraph,
    src: Tuple,
    dst: Tuple,
    edge_loads: Dict[Tuple, int],
) -> Optional[List[Tuple]]:
    def weight_fn(u, v, d):
        load = edge_loads.get((u, v), 0)
        ratio = load / max(d["capacity"], 1)
        return d["base_t"] * (1.0 + config.BPR_ALPHA * ratio ** config.BPR_BETA)

    try:
        return nx.shortest_path(graph, src, dst, weight=weight_fn)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# ---------------------------------------------------------------------------
def _traffic_light_wait(env: simpy.Environment, node: Tuple, light_offsets: Dict) -> float:
    """
    Return simulated wait time for current traffic-light phase at `node`.
    Each intersection has a fixed phase offset so signals are staggered.
    Returns a SimPy timeout event.
    """
    offset = light_offsets[node]
    phase = (env.now + offset) % _PERIOD
    if phase >= config.TRAFFIC_LIGHT_GREEN:
        # Currently RED – wait for next green
        return _PERIOD - phase
    return 0.0


# ---------------------------------------------------------------------------
def _vehicle_process(
    env: simpy.Environment,
    vid: int,
    src: Tuple,
    dst: Tuple,
    spawn: float,
    graph: nx.DiGraph,
    intersections: Dict[Tuple, simpy.Resource],
    light_offsets: Dict[Tuple, float],
    edge_loads: Dict[Tuple, int],
    net: CityNetwork,
    metrics: Dict,
) -> Generator:
    if spawn > env.now:
        yield env.timeout(spawn - env.now)

    path = _dynamic_shortest_path(graph, src, dst, edge_loads)
    if not path or len(path) < 2:
        return

    wait_total = 0.0
    travel_start = env.now
    hops_since_reroute = 0

    step = 0
    while step < len(path) - 1:
        u, v = path[step], path[step + 1]

        if hops_since_reroute >= REROUTE_INTERVAL and u != dst:
            new_path = _dynamic_shortest_path(graph, u, dst, edge_loads)
            if new_path and len(new_path) >= 2:
                path = new_path
                step = 0
                hops_since_reroute = 0
                u, v = path[0], path[1]

        metrics["node_visits"][u] += 1

        # ── Traffic light phase wait (before joining the queue)
        red_wait = _traffic_light_wait(env, u, light_offsets)
        if red_wait > 0:
            wait_total += red_wait
            yield env.timeout(red_wait)

        # ── Intersection queue (capacity-limited)
        t_req = env.now
        with intersections[u].request() as req:
            yield req
            wait_total += env.now - t_req

            metrics["queue_snapshots"].append(
                (env.now, u, len(intersections[u].queue))
            )

            edge_loads[(u, v)] = edge_loads.get((u, v), 0) + 1
            load = intersections[u].count
            tt = net.travel_time(u, v, load)
            yield env.timeout(tt)
            edge_loads[(u, v)] = max(0, edge_loads.get((u, v), 0) - 1)

        step += 1
        hops_since_reroute += 1

    metrics["node_visits"][dst] += 1
    metrics["wait_times"].append(wait_total)
    metrics["travel_times"].append(env.now - travel_start)
    metrics["completed"] += 1


# ---------------------------------------------------------------------------
class SequentialSimulation:
    def __init__(self, network: CityNetwork, num_vehicles: int = config.NUM_VEHICLES):
        self.network = network
        self.num_vehicles = num_vehicles

    def run(self) -> Dict:
        nodes = self.network.nodes
        graph = self.network.graph
        rng = np.random.default_rng(config.SEED)

        vehicle_specs: List[Tuple] = []
        for i in range(self.num_vehicles):
            src = _random_node(rng, nodes)
            dst = _random_node(rng, nodes)
            while dst == src:
                dst = _random_node(rng, nodes)
            spawn = _spawn_time(rng, i, self.num_vehicles)
            vehicle_specs.append((src, dst, spawn))

        # Fixed per-intersection traffic light offset (staggered, deterministic)
        light_rng = np.random.default_rng(config.SEED + 1)
        light_offsets: Dict[Tuple, float] = {
            node: float(light_rng.uniform(0, _PERIOD))
            for node in nodes
        }

        metrics: Dict = {
            "node_visits": defaultdict(int),
            "wait_times": [],
            "travel_times": [],
            "queue_snapshots": [],
            "completed": 0,
        }

        env = simpy.Environment()
        edge_loads: Dict[Tuple, int] = {}

        intersections = {
            node: simpy.Resource(env, capacity=config.INTERSECTION_CAPACITY)
            for node in nodes
        }

        for vid, (src, dst, spawn) in enumerate(vehicle_specs):
            env.process(
                _vehicle_process(
                    env, vid, src, dst, spawn,
                    graph, intersections, light_offsets, edge_loads,
                    self.network, metrics,
                )
            )

        wall_start = time.perf_counter()
        env.run(until=config.SIM_DURATION)
        wall_time = time.perf_counter() - wall_start

        metrics["wall_time"] = wall_time
        metrics["vehicle_specs"] = vehicle_specs
        metrics["light_offsets"] = light_offsets
        return metrics
