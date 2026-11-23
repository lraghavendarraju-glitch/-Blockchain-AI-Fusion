from collections import Counter
from blockai.generator import make_dataset,generate_normal_workflow,random_actor_pool,inject_attack
from blockai.provenance import step_evidence,artifact_digest,sha256,parent_commitment
from blockai.schemas import ATTACKS
import numpy as np


def test_generation_reproducible_and_labeled():
    a=make_dataset(120,42,min_steps=5,max_steps=12)
    b=make_dataset(120,42,min_steps=5,max_steps=12)
    assert [w.to_dict() for w in a] == [w.to_dict() for w in b]
    assert sum(w.label for w in a)==36
    assert len(set(w.workflow_id for w in a))==120
    assert all(5<=len(w.steps)<=12 for w in a)
    assert len({x for wf in a for x in wf.attack_types})==6


def test_normal_verification_and_branching():
    rng=np.random.default_rng(123)
    registry=random_actor_pool(rng)
    good=generate_normal_workflow(0,rng,registry,min_steps=30,max_steps=30)
    assert all(all(step_evidence(s,good)[k] for k in ("seq","acc","int")) for s in good.steps)
    assert any(len(s.parents)>1 for s in good.steps)


def test_all_attack_operators_create_truth_labels():
    rng=np.random.default_rng(9);reg=random_actor_pool(rng)
    good=generate_normal_workflow(8,rng,reg,min_steps=25,max_steps=25)
    for attack in ATTACKS:
        modified=inject_attack(good,attack,np.random.default_rng(8))
        assert modified.label==1
        assert attack in modified.attack_types
        assert modified.attacked_step_ids
        assert good.label==0  # no in-place mutation


def test_label_not_in_cryptographic_evidence():
    rng=np.random.default_rng(17);good=generate_normal_workflow(22,rng,random_actor_pool(rng))
    altered=type(good).from_dict(good.to_dict())
    altered.label=1
    altered.attack_types=["metric_falsification"]
    altered.attacked_step_ids=[altered.steps[0].step_id]
    assert [step_evidence(s,good) for s in good.steps] == [step_evidence(s,altered) for s in altered.steps]


def test_exact_full_scale_event_budget():
    data=make_dataset(1000,seed=42,min_steps=5,max_steps=50)
    assert len(data)==1000
    assert sum(len(w.steps) for w in data)==24000
    assert sum(w.label for w in data)==300
    by_primary=Counter(w.metadata.get('primary_attack') for w in data if w.label)
    assert set(by_primary)==set(ATTACKS)
    assert sorted(by_primary.values())==[50]*6
