from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import defaultdict
import networkx as nx
from .schemas import Workflow
from .provenance import step_evidence
from .ledger import SimulatedLedger


@dataclass
class AuditResult:
    workflow_id: str
    expected_steps: int
    recovered_steps: int
    reconstruction_accuracy: float
    lineage_valid: bool
    missing_step_ids: list[str]
    rejected_step_ids: list[str]
    flagged_step_ids: list[str]
    ground_truth_step_ids: list[str]
    exact_localisation: bool | None
    discrepancies: dict[str, list[str]]

    def to_dict(self) -> dict:
        return asdict(self)


def audit_workflow(wf: Workflow, ledger: SimulatedLedger,
                   step_probabilities: dict[str, float] | None = None,
                   workflow_anomaly: bool = False) -> AuditResult:
    records = ledger.records.get(wf.workflow_id, [])
    record_map = {x["payload"]["step_id"]: x["payload"] for x in records}
    expected = set(wf.expected_step_ids)
    recovered = set(record_map)
    graph = nx.DiGraph()
    graph.add_nodes_from(recovered)
    lineage_valid = ledger.verify_chain()
    discrepancies: dict[str, list[str]] = defaultdict(list)
    for sid, rec in record_map.items():
        for p in rec["parent_ids"]:
            if p in recovered:
                graph.add_edge(p, sid)
            else:
                lineage_valid = False
                discrepancies[sid].append("missing_parent")
    if not nx.is_directed_acyclic_graph(graph):
        lineage_valid = False
    rejected = [a for a in ledger.attempts if a.workflow_id == wf.workflow_id and not a.accepted]
    for attempt in rejected:
        discrepancies[attempt.step_id].extend(attempt.reasons)
    for step in wf.steps:
        ev = step_evidence(step, wf)
        if not ev["model"]:
            discrepancies[step.step_id].append("model_provenance_warning")
    flagged = set(discrepancies)
    if workflow_anomaly and step_probabilities:
        flagged |= {sid for sid, p in step_probabilities.items() if p >= .5}
    # Exact-match is a strict metric; audit rules can flag downstream nodes too.
    return AuditResult(wf.workflow_id, len(expected), len(recovered),
                       len(recovered & expected) / max(1, len(expected)),
                       lineage_valid, sorted(expected - recovered),
                       sorted({a.step_id for a in rejected}), sorted(flagged),
                       sorted(wf.attacked_step_ids),
                       (flagged == set(wf.attacked_step_ids)) if wf.label else None,
                       {k: sorted(set(v)) for k,v in sorted(discrepancies.items())})
