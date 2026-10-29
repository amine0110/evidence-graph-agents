"""Evidence graph schema v1.

A node is one observation (an evidence item), never a world entity. An edge is
a typed relation between two observations. The store is append-only: nothing is
overwritten; belief revision appends a new veracity value with a timestamp.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

SCHEMA_VERSION = "1.0"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO_EVENT = "audio_event"
    SPEECH = "speech"


class EdgeType(str, Enum):
    SAME_EVENT = "same_event"      # symmetric, graded similarity (STS / event coreference)
    SUPPORTS = "supports"          # directed, graded strength (QBAF support)
    CONTRADICTS = "contradicts"    # directed, graded strength (QBAF attack)
    PRECEDES = "precedes"          # directed, from t_obs (Allen / OWL-Time "before")
    DERIVED_FROM = "derived_from"  # directed (W3C PROV-O wasDerivedFrom)


GRADED = {EdgeType.SAME_EVENT, EdgeType.SUPPORTS, EdgeType.CONTRADICTS}
SYMMETRIC = {EdgeType.SAME_EVENT}


@dataclass(frozen=True)
class Source:
    """What produced an observation or an edge."""
    producer: str                   # model or agent name
    version: str                    # model file, checkpoint or code version
    input_ref: Optional[str] = None  # input file or record ID
    offset: Optional[str] = None     # position in the input, e.g. "12.5-14.0s"


@dataclass(frozen=True)
class Node:
    id: str
    modality: Modality
    kind: str                       # domain type: headline, detection, sound_event, ...
    content: dict[str, Any]
    t_obs: tuple[float, float]      # when the observed thing happened (POSIX seconds or clip seconds)
    t_rec: float                    # when the system recorded it; strict past-only reading uses this
    source: Source
    confidence: float
    label: Optional[str] = None     # ground truth, only where the task allows it

    def __post_init__(self):
        _check_unit(self.confidence, "confidence")
        if self.t_obs[0] > self.t_obs[1]:
            raise ValueError(f"node {self.id}: t_obs start after end")


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    type: EdgeType
    source: Source
    confidence: float
    score: Optional[float] = None   # graded value for same_event / supports / contradicts

    def __post_init__(self):
        _check_unit(self.confidence, "confidence")
        if self.type in GRADED:
            if self.score is None:
                raise ValueError(f"{self.type.value} edge needs a score")
            _check_unit(self.score, "score")
        elif self.score is not None:
            raise ValueError(f"{self.type.value} edge takes no score")
        if self.src == self.dst:
            raise ValueError("self-loop")


@dataclass(frozen=True)
class Veracity:
    """One belief value for a node; the newest by `at` is current."""
    node: str
    value: float
    at: float
    rule: str                       # which revision rule produced it
    evidence: tuple[str, ...] = field(default_factory=tuple)  # edge keys that triggered it

    def __post_init__(self):
        _check_unit(self.value, "veracity")


def _check_unit(x, name):
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {x}")


def to_record(obj) -> dict:
    """Serialise a Node, Edge or Veracity to a JSON-ready dict with a record type."""
    record = asdict(obj)
    for key, value in record.items():
        if isinstance(value, Enum):
            record[key] = value.value
    record["record"] = type(obj).__name__.lower()
    record["schema"] = SCHEMA_VERSION
    return record


def from_record(record: dict):
    data = {k: v for k, v in record.items() if k not in ("record", "schema")}
    kind = record["record"]
    if kind == "node":
        data["modality"] = Modality(data["modality"])
        data["t_obs"] = tuple(data["t_obs"])
        data["source"] = Source(**data["source"])
        return Node(**data)
    if kind == "edge":
        data["type"] = EdgeType(data["type"])
        data["source"] = Source(**data["source"])
        return Edge(**data)
    if kind == "veracity":
        data["evidence"] = tuple(data["evidence"])
        return Veracity(**data)
    raise ValueError(f"unknown record type {kind}")
