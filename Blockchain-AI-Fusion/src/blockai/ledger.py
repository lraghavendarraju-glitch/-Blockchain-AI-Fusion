"""Signed, append-only research ledger; NOT a substitute for Hyperledger Fabric."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import json
import os
from pathlib import Path
import time
import requests
from .schemas import Workflow, Step
from .provenance import IdentityVault, commitment_for_step, parent_commitment, step_evidence, sha256


@dataclass
class Attempt:
    workflow_id: str
    step_id: str
    accepted: bool
    reasons: list[str]
    signature_verified: bool
    transaction_id: str | None
    latency_ms: float


class SimulatedLedger:
    """Local functional test double with independent failed-attempt log and block hash chain.

    The class models validation, signing, and audit traversal. Network timings from
    this class are NEVER presented as Hyperledger Fabric measurements.
    """
    def __init__(self, block_size: int = 100, vault: IdentityVault | None = None):
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.block_size = block_size
        self.vault = vault or IdentityVault()
        self.attempts: list[Attempt] = []
        self.records: dict[str, list[dict[str, Any]]] = {}
        self.blocks: list[dict] = []
        self._pending: list[dict] = []
        self._state: dict[str, dict[str, dict]] = {}
        self._trust: dict[str, float] = {}
        self._audit_endorsements: set[tuple[str, str]] = set()

    def set_trust(self, workflow_id: str, trust: float) -> None:
        self._trust[workflow_id] = max(0., min(1., trust))

    def approve_next_step(self, workflow_id: str, step_id: str) -> None:
        self._audit_endorsements.add((workflow_id, step_id))

    def submit(self, wf: Workflow, step: Step) -> Attempt:
        start = time.perf_counter()
        ev = step_evidence(step, wf)
        pseudonym = self.vault.pseudonym(step.actor_id)
        record = commitment_for_step(step, pseudonym)
        signature = step.signature or self.vault.sign(step.actor_id, record)
        signed = self.vault.verify(step.actor_id, record, signature)
        # Authentication signs the proposed transaction; an artifact that was altered
        # after recording its digest remains detectable by output/hash equality.
        state = self._state.setdefault(wf.workflow_id, {})
        reasons = []
        if not ev["seq"]:
            reasons.append("sequence")
        if not ev["acc"]:
            reasons.append("access")
        if not ev["int"]:
            reasons.append("integrity")
        if any(p not in state for p in step.parents):
            reasons.append("missing_parent_commit")
        if not signed:
            reasons.append("signature")
        if self._trust.get(wf.workflow_id, 1.) < .55 and (wf.workflow_id, step.step_id) not in self._audit_endorsements:
            reasons.append("auditor_endorsement_required")
        # The model semantic check is not a chaincode rule: it remains an AI/audit feature.
        accepted = not reasons
        txid = sha256({"record": record, "signature": signature}) if accepted else None
        if accepted:
            entry = {"tx_id": txid, "payload": record, "signature": signature,
                     "actor_id": step.actor_id, "reasons": [], "accepted": True}
            self._pending.append(entry)
            state[step.step_id] = entry
            if len(self._pending) >= self.block_size:
                self._commit_block()
        attempt = Attempt(wf.workflow_id, step.step_id, accepted, sorted(set(reasons)),
                          signed, txid, (time.perf_counter() - start) * 1000)
        self.attempts.append(attempt)
        return attempt

    def process(self, wf: Workflow) -> list[Attempt]:
        return [self.submit(wf, step) for step in wf.steps]

    def _commit_block(self) -> None:
        if not self._pending:
            return
        prev = self.blocks[-1]["block_hash"] if self.blocks else "0" * 64
        index = len(self.blocks)
        tx_ids = [r["tx_id"] for r in self._pending]
        content = {"index": index, "previous_hash": prev, "transaction_ids": tx_ids}
        self.blocks.append({**content, "block_hash": sha256(content), "size": len(tx_ids)})
        for rec in self._pending:
            self.records.setdefault(rec["payload"]["workflow_id"],[]).append(rec)
        self._pending = []

    def finalize(self) -> None:
        self._commit_block()

    def verify_chain(self) -> bool:
        last = "0" * 64
        ids = []
        for i, blk in enumerate(self.blocks):
            content = {"index": blk["index"], "previous_hash": blk["previous_hash"],
                       "transaction_ids": blk["transaction_ids"]}
            if blk["index"] != i or blk["previous_hash"] != last or sha256(content) != blk["block_hash"]:
                return False
            ids.extend(blk["transaction_ids"])
            last = blk["block_hash"]
        accepted_records = [rec for records in self.records.values() for rec in records]
        committed_ids = []
        for rec in accepted_records:
            digest = sha256({"record":rec["payload"],"signature":rec["signature"]})
            if digest != rec["tx_id"]:
                return False
            if not self.vault.verify(rec["actor_id"],rec["payload"],rec["signature"]):
                return False
            committed_ids.append(rec["tx_id"])
        return len(ids)==len(committed_ids) and set(ids)==set(committed_ids) and len(ids)==len(set(ids))

    def statistics(self) -> dict[str, Any]:
        import numpy as np
        total = len(self.attempts)
        accepted = sum(a.accepted for a in self.attempts)
        return {"backend": "simulated_not_fabric", "attempted": total,
                "accepted": accepted, "rejected": total - accepted,
                "acceptance_rate": accepted / max(1,total),
                "blocks": len(self.blocks), "chain_valid": self.verify_chain(),
                "mean_local_processing_ms": float(np.mean([a.latency_ms for a in self.attempts])) if total else 0.0}

    def export(self, path: str | Path) -> None:
        self.finalize()
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        public_records = {key: [{k:v for k,v in rec.items() if k != "actor_id"} for rec in val] for key,val in self.records.items()}
        p.write_text(json.dumps({"blocks": self.blocks, "accepted_records": public_records,
                                 "rejected_attempts": [asdict(a) for a in self.attempts if not a.accepted],
                                 "backend": "simulated_not_fabric"}, indent=2), encoding="utf-8")


class FabricGatewayLedger:
    """Integration with the included authenticated Go REST-to-Fabric Gateway bridge.

    This requires a separately running real Fabric network and a server with an
    enrolled client identity. It cannot silently fall back to a simulated ledger.
    """
    def __init__(self, url: str, token: str | None = None, timeout: int = 30):
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.token = token or os.environ.get("BLOCKAI_GATEWAY_TOKEN", "")
        self.timeout = timeout

    def _request(self, method: str, route: str, body: dict | None = None) -> dict:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        response = self.session.request(method, self.url + route, json=body, headers=headers, timeout=self.timeout)
        try:
            content = response.json()
        except ValueError:
            content = {"raw": response.text}
        if not response.ok:
            raise RuntimeError(f"Fabric gateway HTTP {response.status_code}: {content}")
        return content

    def health(self) -> dict:
        return self._request("GET", "/health")

    def register(self, workflow_id: str, template_id: int) -> dict:
        return self._request("POST", "/workflows", {"workflow_id": workflow_id, "template_id": template_id})

    def submit(self, step: Step, wf: Workflow, vault: IdentityVault | None = None,
               store_private: bool = False) -> dict:
        # Gateway identity holds the X.509 certificate. Client-side actor pseudonym is
        # organizational HMAC and DOES NOT replace Fabric signing.
        vault = vault or IdentityVault()
        record = commitment_for_step(step, vault.pseudonym(step.actor_id))
        if store_private:
            import base64
            from .provenance import canonical
            payload=canonical({"parameters":step.parameters,"salt":step.params_salt})
            record["private_payload_b64"]=base64.b64encode(payload).decode("ascii")
        # Transient private data is submitted by the gateway in the SAME
        # Fabric proposal as SubmitStep (atomic commit or rejection).
        return self._request("POST", "/steps", record)

    def reconstruct(self, workflow_id: str) -> dict:
        return self._request("GET", "/workflows/" + requests.utils.quote(workflow_id, safe=""))

    def replay(self, wf: Workflow, store_private: bool = False) -> dict:
        self.register(wf.workflow_id, wf.template_id)
        results = []
        vault = IdentityVault()
        for step in wf.steps:
            try:
                resp = self.submit(step, wf, vault=vault,store_private=store_private)
                results.append({"step_id": step.step_id, "accepted": True, "result": resp})
            except RuntimeError as exc:
                # Rejected proposed transactions are recorded *off chain*.
                results.append({"step_id": step.step_id, "accepted": False, "error": str(exc)})
        return {"backend": "hyperledger_fabric", "workflow_id": wf.workflow_id, "attempts": results}
