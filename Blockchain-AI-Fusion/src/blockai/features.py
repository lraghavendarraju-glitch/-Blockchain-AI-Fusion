"""Train-only fitted numeric encoders + learned categorical IDs for ProvSeqGAT.

Feature layout: 16 step embedding + 8 transition + 12 parameter/metric
+ 8 role embedding + 4 numeric actor + 8 integrity + 8 temporal = 64.
No ground-truth attack labels, attack types or injected-step IDs are features.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np
from sklearn.preprocessing import StandardScaler
from .schemas import Workflow, STEP_TYPES, ROLES
from .provenance import step_evidence

STEP_IDS = {s: i + 1 for i,s in enumerate(STEP_TYPES)}
ROLE_IDS = {s: i + 1 for i,s in enumerate(ROLES)}


@dataclass
class GraphSample:
    numeric: np.ndarray  # (n,40)
    type_ids: np.ndarray  # (n,)
    role_ids: np.ndarray  # (n,)
    edge_index: np.ndarray  # (2,E)
    global_features: np.ndarray  # (6,)
    label: float
    step_labels: np.ndarray  # (n,)
    workflow_id: str
    step_ids: list[str]


class FeatureEncoder:
    def __init__(self):
        self.numeric_scaler = StandardScaler()
        self.global_scaler = StandardScaler()
        self.transitions: Counter = Counter()
        self.from_totals: Counter = Counter()
        self.fitted = False

    def fit(self, workflows: Sequence[Workflow]) -> "FeatureEncoder":
        if not workflows:
            raise ValueError("Training workflows cannot be empty")
        self.transitions.clear(); self.from_totals.clear()
        for wf in workflows:
            for a,b in zip(wf.steps, wf.steps[1:]):
                self.transitions[(a.step_type,b.step_type)] += 1
                self.from_totals[a.step_type] += 1
        out = [self._raw(wf) for wf in workflows]
        self.numeric_scaler.fit(np.concatenate([x.numeric for x in out], axis=0))
        self.global_scaler.fit(np.stack([x.global_features for x in out]))
        self.fitted = True
        return self

    def transform(self, workflow: Workflow) -> GraphSample:
        if not self.fitted:
            raise RuntimeError("Feature encoder must be fitted exclusively on training data")
        x = self._raw(workflow)
        x.numeric = np.nan_to_num(self.numeric_scaler.transform(x.numeric), nan=0, posinf=0, neginf=0).astype("float32")
        x.global_features = np.nan_to_num(self.global_scaler.transform(x.global_features[None,:])[0], nan=0, posinf=0, neginf=0).astype("float32")
        return x

    def _raw(self, wf: Workflow) -> GraphSample:
        steps = wf.steps
        n = len(steps)
        if not n:
            raise ValueError("Empty workflow")
        positions = {s.step_id: i for i,s in enumerate(steps)}
        outgoing = Counter(p for s in steps for p in s.parents)
        actor_freq = Counter()
        actor_last: dict[str,float] = {}
        depth: dict[str,int] = {}
        prev_time = None
        prev_step = None
        prev_metric = None
        prev_params = None
        evidence = [step_evidence(s,wf) for s in steps]
        numeric: list[list[float]] = []
        for i,s in enumerate(steps):
            ev = evidence[i]
            if prev_step is None:
                logp = 0.0
            else:
                # Add-one smoothing and only TRAINING co-occurrence counts.
                num = 1 + self.transitions.get((prev_step.step_type,s.step_type),0)
                den = len(STEP_TYPES) + self.from_totals.get(prev_step.step_type,0)
                logp = math.log(num / max(1,den))
            branch_depth = max([depth.get(p,0) for p in s.parents], default=0) + 1
            depth[s.step_id] = branch_depth
            # 8 transition/structural features
            transition = [logp, float(i>0 and steps[i-1].step_id not in s.parents),
                          float(not ev["seq"]), min(branch_depth,50)/50.,
                          float(len(s.parents)), float(outgoing[s.step_id]),
                          float(i == 0), float(i == n-1)]
            p = s.parameters
            lr = float(p.get("learning_rate", 1e-3))
            fc = float(p.get("feature_count", 0))
            batch = float(p.get("batch_size", 32))
            prior = prev_params or p
            plr = float(prior.get("learning_rate", lr))
            pfc = float(prior.get("feature_count", fc))
            pbatch = float(prior.get("batch_size", batch))
            changed = sum(p.get(k) != prior.get(k) for k in set(p)|set(prior)) if prev_params else 0
            norm_change = abs(lr-plr)/(abs(plr)+1e-6) + abs(fc-pfc)/(abs(pfc)+1) + abs(batch-pbatch)/(abs(pbatch)+1)
            metric = float(s.metric) if s.metric is not None else None
            metric_delta = 0. if metric is None or prev_metric is None else metric-prev_metric
            # 12 parameter and metric features
            params = [math.log10(max(lr,1e-10)), math.log2(max(batch,1)),
                      fc/100., float(changed), norm_change,
                      metric_delta, 0. if metric is None else metric - .78,
                      0. if metric is None else metric,
                      (lr-plr)/(abs(plr)+1e-6), float(metric is not None),
                      float(s.artifact_version - wf.version),
                      float(s.step_type in {"evaluate","monitor"} and metric is None)]
            # Four non-categorical actor features (8-dimensional role is learned).
            is_new = s.actor_id not in actor_freq
            actor = [float(ev["acc"]), math.log1p(actor_freq[s.actor_id]),
                     math.log1p(max(0., s.timestamp - actor_last.get(s.actor_id,s.timestamp))),
                     float(is_new)]
            actor_freq[s.actor_id] += 1; actor_last[s.actor_id] = s.timestamp
            # Eight ledger-verifiable/policy and provenance semantic features.
            # Signature is verified by the ledger adapter, defaults to 1 for
            # synthetically generated and locally signed proposals.
            integrity = [float(ev["seq"]), float(ev["acc"]), float(ev["int"]),
                         float(ev["link"]), float(s.artifact_version == wf.version),
                         float(ev["model"]), float(ev["report"]),
                         1.0]
            d = max(0., s.timestamp - (prev_time if prev_time is not None else s.timestamp))
            # 8 temporal features, including chronological consistency.
            hr = (s.timestamp % 86400) / 86400 * 2*math.pi
            temporal = [math.log1p(d), math.log1p(max(0,s.duration)),
                        (math.log1p(s.duration)-1.5)/1.,
                        math.sin(hr), math.cos(hr), float(prev_time is None or s.timestamp >= prev_time),
                        i / max(1,n-1), math.log1p(branch_depth)]
            row = transition + params + actor + integrity + temporal
            if len(row) != 40:
                raise AssertionError(f"Numeric width {len(row)} != 40")
            numeric.append(row)
            prev_time = s.timestamp
            prev_step = s
            prev_params = p
            if metric is not None:
                prev_metric = metric
        edges = [(positions[p], i) for i,s in enumerate(steps) for p in s.parents if p in positions]
        edge_index = np.array(edges,dtype=np.int64).T if edges else np.empty((2,0),dtype=np.int64)
        failure_fraction = sum(any(not e[k] for k in ("seq","acc","int")) for e in evidence) / n
        glob = np.array([n, len({s.actor_id for s in steps}), len(edges)/max(1,n),
                         steps[-1].timestamp-steps[0].timestamp,
                         failure_fraction, wf.version], dtype=np.float32)
        attacked = set(wf.attacked_step_ids)
        return GraphSample(np.asarray(numeric,dtype=np.float32),
                           np.asarray([STEP_IDS.get(s.step_type,0) for s in steps],dtype=np.int64),
                           np.asarray([ROLE_IDS.get(s.role,0) for s in steps],dtype=np.int64),
                           edge_index, glob, float(wf.label),
                           np.array([float(s.step_id in attacked) for s in steps],dtype=np.float32),
                           wf.workflow_id, [s.step_id for s in steps])

    def save(self, path: str) -> None:
        import pickle
        from pathlib import Path
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        with open(p,"wb") as f: pickle.dump(self,f)

    @staticmethod
    def load(path: str) -> "FeatureEncoder":
        import pickle
        with open(path,"rb") as f: return pickle.load(f)
