"""Public serialisation schema; attack labels are deliberately separate from features."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import json
import gzip
from pathlib import Path

STEP_TYPES = (
    "ingest", "validate", "preprocess", "feature_engineer", "train", "tune",
    "evaluate", "register", "deploy", "monitor", "retrain", "audit",
)
ROLES = (
    "data_engineer", "data_scientist", "ml_engineer", "reviewer",
    "mlops_engineer", "auditor", "administrator", "external_collaborator",
)
ATTACKS = (
    "data_tampering", "feature_manipulation", "model_replacement",
    "metric_falsification", "unauthorized_access", "workflow_reordering",
)
SEVERITY = {
    "data_tampering": 8.1, "feature_manipulation": 7.4,
    "model_replacement": 9.0, "metric_falsification": 6.5,
    "unauthorized_access": 7.8, "workflow_reordering": 6.1,
}
# Static policy that permits repeated operations, but requires a certified role.
ROLE_POLICY = {
    "ingest": {"data_engineer", "administrator"},
    "validate": {"data_engineer", "reviewer", "administrator"},
    "preprocess": {"data_engineer", "data_scientist", "administrator"},
    "feature_engineer": {"data_scientist", "ml_engineer", "administrator"},
    "train": {"data_scientist", "ml_engineer", "administrator"},
    "tune": {"data_scientist", "ml_engineer", "administrator"},
    "evaluate": {"reviewer", "data_scientist", "administrator"},
    "register": {"ml_engineer", "administrator"},
    "deploy": {"mlops_engineer", "administrator"},
    "monitor": {"mlops_engineer", "auditor", "administrator"},
    "retrain": {"ml_engineer", "administrator"},
    "audit": {"auditor", "administrator"},
}

@dataclass
class Step:
    step_id: str
    step_type: str
    actor_id: str
    role: str
    parents: list[str]
    timestamp: float
    duration: float
    parameters: dict[str, Any]
    metric: float | None
    artifact: dict[str, Any]
    input_hash: str
    output_hash: str
    report_hash: str
    model_hash: str
    artifact_version: int
    # index is the declared execution index, NOT the topological order.
    execution_index: int
    # Signals are computed from evidence, not the injected label.
    signature: str = ""
    params_salt: str = ""

@dataclass
class Workflow:
    workflow_id: str
    template_id: int
    version: int
    steps: list[Step]
    expected_step_ids: list[str]
    label: int = 0
    attack_types: list[str] = field(default_factory=list)
    attacked_step_ids: list[str] = field(default_factory=list)
    provenance_source: str = "synthetic"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "Workflow":
        rec = dict(obj)
        rec["steps"] = [Step(**s) for s in rec.get("steps", [])]
        return cls(**rec)

    def step_map(self) -> dict[str, Step]:
        return {s.step_id: s for s in self.steps}


def save_workflows(workflows: list[Workflow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        for wf in workflows:
            f.write(json.dumps(wf.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")


def load_workflows(path: str | Path, limit: int | None = None) -> list[Workflow]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    workflows = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                workflows.append(Workflow.from_dict(json.loads(line)))
            if limit and len(workflows) >= limit:
                break
    return workflows
