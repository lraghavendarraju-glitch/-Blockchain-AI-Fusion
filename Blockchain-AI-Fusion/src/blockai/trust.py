"""Trust score, temporal actor-history maintenance, and separate risk classification."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from .schemas import Workflow
from .provenance import step_evidence

TRUST_WEIGHTS=(.60,.25,.15)


def integrity_fraction(wf: Workflow) -> float:
    return sum(all(step_evidence(s,wf)[k] for k in ("seq","acc","int")) for s in wf.steps) / max(1,len(wf.steps))


def workflow_trust(anomaly_prob: float, verified_fraction: float,
                   mean_actor_trust: float, weights: tuple[float,float,float] = TRUST_WEIGHTS) -> float:
    if any(x<0 or x>1 for x in (anomaly_prob,verified_fraction,mean_actor_trust)):
        raise ValueError("Trust components must be in [0,1]")
    if abs(sum(weights)-1)>1e-8 or any(w < 0 for w in weights):
        raise ValueError("Trust weights must be nonnegative and sum to 1")
    a,b,c=weights
    return max(0.,min(1.,a*(1-anomaly_prob)+b*verified_fraction+c*mean_actor_trust))


def classify_risk(anomaly_prob: float, theta1: float = .46, theta2: float = .8) -> str:
    if anomaly_prob < theta1:
        return "normal"
    if anomaly_prob < theta2:
        return "suspicious"
    return "compromised"


class ChronologicalTrustTracker:
    def __init__(self, initial_trust: float = .8):
        self.initial_trust=initial_trust
        self.history: dict[str,list[float]] = defaultdict(list)

    def score(self, wf: Workflow, anomaly_prob: float,
              verified_fraction: float | None = None,
              weights: tuple[float,float,float] = TRUST_WEIGHTS) -> dict:
        actors={step.actor_id for step in wf.steps}
        actor_mean=(sum(sum(self.history[a])/len(self.history[a]) if self.history[a] else self.initial_trust
                        for a in actors)/max(1,len(actors)))
        verified=integrity_fraction(wf) if verified_fraction is None else verified_fraction
        trust=workflow_trust(anomaly_prob,verified,actor_mean,weights)
        # Update AFTER prediction, so no future/test outcome enters the current score.
        for a in actors:
            self.history[a].append(trust)
        return {"workflow_id":wf.workflow_id,"anomaly_probability":anomaly_prob,
                "integrity_fraction":verified,"mean_historical_actor_trust":actor_mean,
                "trust_score":trust,"requires_auditor":trust<.55,
                "actor_count":len(actors)}


def score_batch_chronologically(workflows: list[Workflow], predictions: list[dict],
                                weights: tuple[float,float,float] = TRUST_WEIGHTS) -> list[dict]:
    lookup={p["workflow_id"]:p for p in predictions}
    # Workflows are processed by their first observed timestamp, not by their labels.
    ordering=sorted(workflows,key=lambda w:w.steps[0].timestamp)
    tracker=ChronologicalTrustTracker()
    return [tracker.score(w,float(lookup[w.workflow_id]["anomaly_probability"]),weights=weights)
            for w in ordering]
