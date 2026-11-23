import os
from pathlib import Path
import pytest
from blockai.generator import make_dataset
from blockai.provenance import IdentityVault,EncryptedArtifactStore,sha256
from blockai.ledger import SimulatedLedger
from blockai.audit import audit_workflow
from blockai.trust import workflow_trust,classify_risk,ChronologicalTrustTracker


def test_ecdsa_sign_verify():
    signer=IdentityVault()
    obj={"record":1,"workflow":"w"}
    signature=signer.sign("actor",obj)
    assert signer.verify("actor",obj,signature)
    assert not signer.verify("actor",{"record":2},signature)
    assert signer.pseudonym("actor")!=signer.pseudonym("other")


def test_encrypted_store_roundtrip(tmp_path):
    secret=os.urandom(32)
    store=EncryptedArtifactStore(tmp_path,secret)
    record={"foo":"bar","feature_set":[.1,.2]}
    digest=store.write(record)
    assert store.read(digest)==record
    assert b'foo' not in (tmp_path/(digest+'.enc')).read_bytes()
    with open(tmp_path/(digest+'.enc'),"ab") as f:f.write(b'evil')
    with pytest.raises(Exception):store.read(digest)


def test_simulated_ledger_chain_and_audit():
    samples=make_dataset(48,seed=33,min_steps=5,max_steps=18)
    ledger=SimulatedLedger(block_size=20)
    for wf in samples:ledger.process(wf)
    ledger.finalize()
    assert ledger.verify_chain()
    assert len(ledger.attempts)==sum(len(w.steps) for w in samples)
    assert any(not a.accepted for a in ledger.attempts)
    good=next(x for x in samples if x.label==0)
    audited=audit_workflow(good,ledger)
    assert audited.reconstruction_accuracy==1
    assert audited.lineage_valid
    ledger.blocks[0]['previous_hash']='corrupted'
    assert not ledger.verify_chain()


def test_trust_and_auditor_feedback():
    assert workflow_trust(0.,1.,.8)==pytest.approx(.97)
    assert classify_risk(.6)=="suspicious"
    wf=next(w for w in make_dataset(12,seed=4) if w.label==0)
    ledger=SimulatedLedger()
    ledger.set_trust(wf.workflow_id,.2)
    rejected=ledger.submit(wf,wf.steps[0])
    assert rejected.accepted is False
    assert "auditor_endorsement_required" in rejected.reasons
    ledger.approve_next_step(wf.workflow_id,wf.steps[0].step_id)
    approved=ledger.submit(wf,wf.steps[0])
    assert approved.accepted
    tracker=ChronologicalTrustTracker()
    one=tracker.score(wf,.6)
    assert one['mean_historical_actor_trust']==pytest.approx(.8)


def test_committed_vs_pending_and_record_tamper():
    wf=next(w for w in make_dataset(12,seed=58) if w.label==0)
    ledger=SimulatedLedger(block_size=100)
    ledger.process(wf)
    assert wf.workflow_id not in ledger.records  # only pending, not yet committed
    ledger.finalize()
    assert ledger.verify_chain()
    ledger.records[wf.workflow_id][0]['payload']['step_type']='malicious_change'
    assert not ledger.verify_chain()
