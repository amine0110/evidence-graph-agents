"""Append-only JSONL evidence graph with the queries the reasoning agent needs."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from .schema import Edge, EdgeType, Node, Veracity, from_record, to_record


class EvidenceGraph:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.nodes: dict[str, Node] = {}
        self.out_edges: dict[str, list[Edge]] = defaultdict(list)
        self.in_edges: dict[str, list[Edge]] = defaultdict(list)
        self.beliefs: dict[str, list[Veracity]] = defaultdict(list)
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._apply(from_record(json.loads(line)))

    # --- writing ---------------------------------------------------------

    def add(self, obj):
        """Add a Node, Edge or Veracity; it is persisted before it is applied."""
        self._validate(obj)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(to_record(obj), ensure_ascii=False) + "\n")
        self._apply(obj)
        return obj

    def _validate(self, obj):
        if isinstance(obj, Node) and obj.id in self.nodes:
            raise ValueError(f"node {obj.id} already exists; nodes are immutable")
        if isinstance(obj, Edge):
            for end in (obj.src, obj.dst):
                if end not in self.nodes:
                    raise ValueError(f"edge refers to unknown node {end}")
        if isinstance(obj, Veracity) and obj.node not in self.nodes:
            raise ValueError(f"veracity for unknown node {obj.node}")

    def _apply(self, obj):
        if isinstance(obj, Node):
            self.nodes[obj.id] = obj
        elif isinstance(obj, Edge):
            self.out_edges[obj.src].append(obj)
            self.in_edges[obj.dst].append(obj)
        else:
            self.beliefs[obj.node].append(obj)

    # --- reading ---------------------------------------------------------

    def neighbours(self, node_id: str, types: Optional[Iterable[EdgeType]] = None,
                   recorded_before: Optional[float] = None) -> list[tuple[Edge, Node]]:
        """Edges touching node_id in either direction, with the node at the other end.

        `recorded_before` keeps only neighbours whose t_rec is strictly earlier:
        the past-only rule of C2.
        """
        wanted = set(types) if types else None
        touching = [(e, e.dst) for e in self.out_edges[node_id]]
        touching += [(e, e.src) for e in self.in_edges[node_id]]
        return [
            (edge, self.nodes[other])
            for edge, other in touching
            if (wanted is None or edge.type in wanted)
            and (recorded_before is None or self.nodes[other].t_rec < recorded_before)
        ]

    def in_window(self, start: float, end: float, modality=None) -> list[Node]:
        """Nodes whose observation interval overlaps [start, end]."""
        return sorted(
            (n for n in self.nodes.values()
             if n.t_obs[0] <= end and n.t_obs[1] >= start
             and (modality is None or n.modality == modality)),
            key=lambda n: n.t_obs[0],
        )

    def records(self) -> list:
        """Every node, then every edge, then every belief: an order that can be re-added."""
        edges = [e for out in self.out_edges.values() for e in out]
        beliefs = [v for history in self.beliefs.values() for v in history]
        return [*self.nodes.values(), *edges, *beliefs]

    def veracity(self, node_id: str) -> Optional[Veracity]:
        history = self.beliefs.get(node_id)
        return max(history, key=lambda v: v.at) if history else None
