import networkx as nx
import numpy as np
from typing import List, Optional, Tuple


class CityNetwork:
    """
    City road network modelled as a directed weighted graph.
    Nodes = intersections (i, j), edges = road segments.
    Edge weight = base travel time (minutes) via speed limit.
    Congestion uses the BPR (Bureau of Public Roads) function.
    """

    def __init__(self, grid_size: int = 25, seed: int = 42):
        self.grid_size = grid_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.graph = self._build()

    def _build(self) -> nx.DiGraph:
        n = self.grid_size
        G = nx.DiGraph()

        for i in range(n):
            for j in range(n):
                G.add_node((i, j), x=i, y=j)

        for i in range(n):
            for j in range(n):
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        # Major arterials every 5 blocks — faster, higher capacity
                        if i % 5 == 0 or j % 5 == 0:
                            speed = float(self.rng.choice([60, 70, 80]))
                            cap = int(self.rng.integers(30, 60))
                        else:
                            speed = float(self.rng.choice([30, 40, 50]))
                            cap = int(self.rng.integers(10, 25))

                        dist = 0.3  # km per block
                        base_t = (dist / speed) * 60.0  # minutes
                        G.add_edge(
                            (i, j), (ni, nj),
                            speed=speed, distance=dist,
                            base_t=base_t, capacity=cap,
                        )
        return G

    def travel_time(self, u: Tuple, v: Tuple, load: int = 0) -> float:
        """BPR congestion function: t = t0 * (1 + alpha*(v/c)^beta)."""
        e = self.graph[u][v]
        ratio = load / max(e["capacity"], 1)
        return e["base_t"] * (1.0 + 0.15 * ratio ** 4)

    def shortest_path(self, src: Tuple, dst: Tuple) -> Optional[List[Tuple]]:
        try:
            return nx.shortest_path(self.graph, src, dst, weight="base_t")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    @property
    def nodes(self) -> List[Tuple]:
        return list(self.graph.nodes())

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()
