from __future__ import annotations
from copy import deepcopy
from collections import Counter
from typing import Sequence
import math
import numpy as np
from .schemas import Workflow, Step, STEP_TYPES, ROLES, ROLE_POLICY, ATTACKS
from .provenance import sha256, parent_commitment, report_digest


def random_actor_pool(rng: np.random.Generator, n: int = 312) -> dict[str, str]:
    # A reproducible pool of known roles. Admins participate in multi-role workflows.
    roles = list(ROLES) * (n // len(ROLES) + 1)
    rng.shuffle(roles)
    return {f"ACT_{i:04d}": roles[i] for i in range(n)}


def _types_for_template(template_id: int, n: int) -> list[str]:
    """20 distinct CRISP-DM-style cyclic templates with fixed lifecycle anchors."""
    stages = list(STEP_TYPES)
    variants = [
        ["validate", "preprocess"], ["feature_engineer", "tune"],
        ["evaluate", "audit"], ["monitor", "retrain"],
        ["train", "evaluate"],
    ]
    block = stages.copy()
    # Vary branch-related/retraining positions without losing lifecycle anchor stages.
    if template_id % 5:
        insert = variants[template_id % 5]
        block = block[:7] + insert + block[7:]
    rotation = (template_id // 5) % 4
    if rotation:
        # Rotations only inside independently repeatable preprocessing/training sub-block.
        sub = block[2:7]
        block[2:7] = sub[rotation:] + sub[:rotation]
    return [block[i % len(block)] for i in range(n)]


def generate_normal_workflow(index: int, rng: np.random.Generator, registry: dict[str, str],
                             min_steps: int = 5, max_steps: int = 50,
                             forced_length: int | None = None) -> Workflow:
    n = int(forced_length) if forced_length is not None else max(min_steps, min(max_steps, int(round(rng.normal(24, 7)))))
    template_id = index % 20
    version = int(rng.integers(1, 6))
    wf_id = f"WF_{index:06d}"
    counts = rng.choice([1, 2, 3, 4, 5, 6, 7, 8], p=[.4, .175, .175, .1, .06, .04, .03, .02])
    # Zipf activity allocation, using rank weights.
    actor_ids = list(registry)
    weights = np.arange(1, len(actor_ids) + 1, dtype=float) ** -1.1
    weights = weights / weights.sum()
    admins = [a for a in actor_ids if registry[a] == "administrator"]
    admin = str(rng.choice(admins))
    chosen = [admin]
    if counts > 1:
        candidates = [a for a in actor_ids if a != admin]
        p = np.array([weights[int(a.split("_")[-1])] for a in candidates]); p /= p.sum()
        chosen.extend(str(x) for x in rng.choice(candidates, size=int(counts)-1, replace=False, p=p))
    step_types = _types_for_template(template_id, n)
    steps: list[Step] = []
    ts = 1720000000.0 + index * 30
    baseline_lr = float(np.exp(rng.normal(-7, .4)))
    feature_count = int(rng.integers(24, 90))
    prev_metric = 0.75 + float(rng.normal(0, .02))
    model_hash = sha256({"wf": wf_id, "init_model": True})
    for i, step_type in enumerate(step_types):
        step_id = f"{wf_id}:S{i:03d}"
        if i == 0:
            parents: list[str] = []
        else:
            parent_indices = [i - 1]
            if i >= 3 and rng.random() < 0.25:
                parent_indices.append(int(rng.integers(max(0, i-4), i-1)))
            parents = sorted({steps[j].step_id for j in parent_indices})
        role_candidates = [a for a in chosen if registry[a] in ROLE_POLICY[step_type]]
        actor = str(rng.choice(role_candidates)) if role_candidates else admin
        role = registry[actor]
        ts += float(rng.lognormal(3.2, .9))
        duration = float(rng.lognormal(1.0 + (step_type in {"train", "tune"}), .5))
        baseline_lr *= float(rng.normal(1, .05))
        feature_count = max(1, feature_count + int(rng.choice([-2, -1, 0, 0, 0, 1, 2])))
        params = {"learning_rate": round(baseline_lr, 9), "feature_count": feature_count,
                  "batch_size": int(rng.choice([16, 32, 64, 128]))}
        metric = None
        if step_type in {"train", "tune", "evaluate", "monitor", "retrain"}:
            prev_metric = max(.02, min(.99, prev_metric + float(rng.normal(.002 if rng.random() < .7 else -.001, .004))))
            metric = round(prev_metric, 5)
        if step_type in {"train", "retrain", "tune"}:
            model_hash = sha256({"wf": wf_id, "train_step": step_id, "params": params})
        parent_digests = [(pid, next(s.output_hash for s in steps if s.step_id == pid)) for pid in parents]
        inp_hash = parent_commitment(parent_digests)
        artifact = {"workflow": wf_id, "step_id": step_id, "type": step_type,
                    "data_sample": round(float(rng.normal()), 6), "feature_count": feature_count,
                    "model_hash": model_hash}
        output_hash = sha256(artifact)
        step = Step(step_id, step_type, actor, role, parents, ts, duration, params, metric, artifact,
                    inp_hash, output_hash, "", model_hash, version, i)
        step.params_salt = rng.bytes(16).hex()
        step.report_hash = report_digest(step)
        steps.append(step)
    return Workflow(wf_id, template_id, version, steps,
                    [s.step_id for s in steps], metadata={"generator_seeded": True})


def rechain_from(wf: Workflow, start: int = 0) -> None:
    """Signed insider can commit a modified but internally consistent downstream chain."""
    mapping = wf.step_map()
    for step in wf.steps[start:]:
        step.input_hash = parent_commitment([(p, mapping[p].output_hash) for p in step.parents if p in mapping])
        step.output_hash = sha256(step.artifact)
        step.report_hash = report_digest(step)


def inject_attack(normal: Workflow, attack: str, rng: np.random.Generator) -> Workflow:
    wf = deepcopy(normal)
    if attack not in ATTACKS:
        raise ValueError(f"Unknown attack: {attack}")
    candidate_idx = [i for i, s in enumerate(wf.steps) if s.step_type in {"register", "deploy"}]
    idx = int(rng.choice(candidate_idx if attack == "model_replacement" and candidate_idx else range(len(wf.steps))))
    s = wf.steps[idx]
    if attack == "data_tampering":
        s.artifact["data_sample"] = float(s.artifact["data_sample"]) + float(rng.uniform(.5, 3))
        # Post-commit artifact corruption: claimed artifact hash remains unchanged.
    elif attack == "feature_manipulation":
        s.artifact["feature_count"] = max(0, int(s.artifact["feature_count"]) - int(rng.integers(1, 4)))
        s.parameters["feature_count"] = s.artifact["feature_count"]
        rechain_from(wf, idx)  # authorized, re-hashed manipulation
    elif attack == "model_replacement":
        forged = sha256({"forged_model": wf.workflow_id, "step": s.step_id})
        s.model_hash = forged
        s.artifact["model_hash"] = forged
        rechain_from(wf, idx)  # model provenance now inconsistent despite rehashing
    elif attack == "metric_falsification":
        s.metric = float(s.metric if s.metric is not None else .7) + float(rng.uniform(.02, .08))
        if rng.random() < 0.5:
            # Insider falsification may not violate any cryptographic rule.
            s.report_hash = report_digest(s)
    elif attack == "unauthorized_access":
        s.role = "external_collaborator"  # a real identity with an insufficient role
        s.actor_id = "ACT_UNPRIVILEGED"
    elif attack == "workflow_reordering":
        if len(wf.steps) >= 2:
            k = max(1, idx)
            wf.steps[k - 1], wf.steps[k] = wf.steps[k], wf.steps[k - 1]
            s = wf.steps[k - 1]
    wf.label = 1
    wf.attack_types = sorted(set(wf.attack_types + [attack]))
    wf.attacked_step_ids = sorted(set(wf.attacked_step_ids + [s.step_id]))
    return wf


def make_dataset(n_workflows: int = 10000, seed: int = 42,
                 attack_fraction: float = .30, min_steps: int = 5, max_steps: int = 50) -> list[Workflow]:
    if n_workflows < 2 or not 0 <= attack_fraction <= 1 or min_steps > max_steps:
        raise ValueError("Invalid workflow generation settings")
    rng = np.random.default_rng(seed)
    registry = random_actor_pool(rng)
    n_attacks = round(n_workflows * attack_fraction)
    count = n_attacks // len(ATTACKS)
    attacks = [a for a in ATTACKS for _ in range(count)]
    attacks += list(ATTACKS[:n_attacks - len(attacks)])
    assignments: list[str | None] = [None] * (n_workflows - n_attacks) + attacks
    rng.shuffle(assignments)
    lengths = None
    if n_workflows >= 1000 and min_steps <= 24 <= max_steps:
        # Match the reference 240,000-event target for the 10,000-trace experiment.
        # Keep individual lengths within the paper's truncated-normal bounds.
        lengths = np.clip(np.rint(rng.normal(24,7,size=n_workflows)).astype(int),min_steps,max_steps)
        delta = 24*n_workflows - int(lengths.sum())
        while delta:
            eligible = np.flatnonzero(lengths < max_steps if delta>0 else lengths>min_steps)
            take = min(abs(delta),len(eligible))
            chosen = rng.choice(eligible,size=take,replace=False)
            lengths[chosen] += 1 if delta>0 else -1
            delta -= take if delta>0 else -take
    results = []
    for i, assigned in enumerate(assignments):
        wf = generate_normal_workflow(i, rng, registry, min_steps, max_steps,
                                      forced_length=int(lengths[i]) if lengths is not None else None)
        if assigned is not None:
            wf = inject_attack(wf, assigned, rng)
            wf.metadata["primary_attack"] = assigned
            # The 30% multi-stage subset has 2 or 3 affected steps, potentially multiple categories.
            if rng.random() < .30:
                for _ in range(int(rng.integers(1, 3))):
                    wf = inject_attack(wf, str(rng.choice(ATTACKS)), rng)
        results.append(wf)
    return results
