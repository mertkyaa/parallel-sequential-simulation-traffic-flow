from typing import Dict, List, Set, Tuple
import networkx as nx
from src.network import CityNetwork


class SectorPartitioner:
    """
    Divides the grid network into num_x * num_y rectangular sectors.
    Boundary nodes are those whose edges cross sector borders.
    Partitioning is done by spatial proximity (minimum cut implied by grid structure).
    """

    def __init__(self, network: CityNetwork, num_x: int = 2, num_y: int = 2):
        self.network = network
        self.num_x = num_x
        self.num_y = num_y
        self.num_sectors = num_x * num_y
        self.n = network.grid_size

        self.node_to_sector: Dict[Tuple, int] = {}
        self.sectors: Dict[int, Set[Tuple]] = {i: set() for i in range(self.num_sectors)}
        self.boundary_nodes: Set[Tuple] = set()

        self._assign()
        self._find_boundaries()

    # ------------------------------------------------------------------
    def _sector_id(self, node: Tuple) -> int:
        i, j = node
        sx = min(int(i // (self.n / self.num_x)), self.num_x - 1)
        sy = min(int(j // (self.n / self.num_y)), self.num_y - 1)
        return sx * self.num_y + sy

    def _assign(self) -> None:
        for node in self.network.nodes:
            sid = self._sector_id(node)
            self.node_to_sector[node] = sid
            self.sectors[sid].add(node)

    def _find_boundaries(self) -> None:
        for u, v in self.network.graph.edges():
            if self.node_to_sector[u] != self.node_to_sector[v]:
                self.boundary_nodes.add(u)
                self.boundary_nodes.add(v)

    # ------------------------------------------------------------------
    def sector_of(self, node: Tuple) -> int:
        return self.node_to_sector[node]

    def subgraph(self, sector_id: int) -> nx.DiGraph:
        return self.network.graph.subgraph(self.sectors[sector_id]).copy()

    def cross_sector_edges(self) -> int:
        return sum(
            1 for u, v in self.network.graph.edges()
            if self.node_to_sector[u] != self.node_to_sector[v]
        )

    def partition_report(self) -> Dict:
        return {
            "num_sectors": self.num_sectors,
            "nodes_per_sector": {sid: len(nodes) for sid, nodes in self.sectors.items()},
            "boundary_nodes": len(self.boundary_nodes),
            "cross_sector_edges": self.cross_sector_edges(),
            "total_edges": self.network.num_edges,
        }
