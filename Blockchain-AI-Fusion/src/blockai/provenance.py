from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
import base64
import hashlib
import hmac
import json
from pathlib import Path
import os

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidSignature

from .schemas import Step, Workflow, ROLE_POLICY


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(obj: Any) -> str:
    if isinstance(obj, bytes):
        payload = obj
    else:
        payload = canonical(obj)
    return hashlib.sha256(payload).hexdigest()


def parent_commitment(parents: list[tuple[str, str]]) -> str:
    """Unambiguous commitment for *all* DAG parents, including fork-join nodes."""
    return sha256(sorted([[pid, digest] for pid, digest in parents]))


def artifact_digest(step: Step) -> str:
    return sha256(step.artifact)


def report_digest(step: Step) -> str:
    return sha256({"metric": step.metric, "step_id": step.step_id})


def step_evidence(step: Step, wf: Workflow) -> dict[str, Any]:
    """Independent, label-free rule evidence from a proposed workflow execution."""
    m = wf.step_map()
    positions = {s.step_id: i for i, s in enumerate(wf.steps)}
    valid_parents = all(p in m and positions[p] < positions[step.step_id] for p in step.parents)
    parent_hashes = [(p, m[p].output_hash) for p in step.parents if p in m]
    linkage = len(parent_hashes) == len(step.parents) and step.input_hash == parent_commitment(parent_hashes)
    auth = step.role in ROLE_POLICY.get(step.step_type, set())
    artifact_ok = step.output_hash == artifact_digest(step)
    report_ok = step.report_hash == report_digest(step)
    # Missing training evidence is a semantic warning, not cryptographic proof.
    if step.step_type in {"register", "deploy"}:
        prior_models = [s.model_hash for s in wf.steps[:positions[step.step_id]] if s.step_type in {"train", "retrain", "tune"}]
        model_ok = bool(prior_models) and step.model_hash in prior_models
    else:
        model_ok = True
    return {
        "seq": bool(valid_parents and step.execution_index == positions[step.step_id]),
        "acc": bool(auth),
        "int": bool(linkage and artifact_ok and report_ok),
        "link": bool(linkage),
        "artifact": bool(artifact_ok),
        "report": bool(report_ok),
        "model": bool(model_ok),
        "signature": None,  # set by the ledger after cryptographic verification
    }


def commitment_for_step(step: Step, pseudonym: str) -> dict[str, Any]:
    return {
        "workflow_id": step.step_id.split(":")[0],
        "step_id": step.step_id,
        "step_type": step.step_type,
        "parent_ids": list(step.parents),
        "input_hash": step.input_hash,
        "output_hash": step.output_hash,
        "artifact_hash": sha256(step.artifact),
        "params_commitment": sha256({"parameters": step.parameters, "salt": step.params_salt}),
        "metric_commitment": step.report_hash,
        "actor_pseudonym": pseudonym,
        "timestamp": int(step.timestamp),
        "execution_index": step.execution_index,
        "artifact_version": step.artifact_version,
        "model_hash": step.model_hash,
    }


class IdentityVault:
    """In-memory demo signing keys; do not persist plaintext private keys in shared traces."""
    def __init__(self, pseudonym_key: bytes | None = None):
        self._keys: dict[str, ec.EllipticCurvePrivateKey] = {}
        self._pseudonym_key = pseudonym_key or os.urandom(32)

    def pseudonym(self, actor_id: str) -> str:
        return hmac.new(self._pseudonym_key, actor_id.encode(), hashlib.sha256).hexdigest()

    def key(self, actor_id: str) -> ec.EllipticCurvePrivateKey:
        if actor_id not in self._keys:
            self._keys[actor_id] = ec.generate_private_key(ec.SECP256R1())
        return self._keys[actor_id]

    def sign(self, actor_id: str, payload: dict) -> str:
        signature = self.key(actor_id).sign(canonical(payload), ec.ECDSA(hashes.SHA256()))
        return base64.b64encode(signature).decode("ascii")

    def verify(self, actor_id: str, payload: dict, signature: str) -> bool:
        try:
            self.key(actor_id).public_key().verify(base64.b64decode(signature), canonical(payload), ec.ECDSA(hashes.SHA256()))
            return True
        except (InvalidSignature, ValueError, TypeError):
            return False


class EncryptedArtifactStore:
    """AES-256-GCM with random per-object nonce; key is supplied out of band."""
    def __init__(self, root: str | Path, key: bytes):
        if len(key) != 32:
            raise ValueError("AES-256-GCM requires a 32-byte key")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.aead = AESGCM(key)

    def write(self, artifact: dict[str, Any]) -> str:
        payload = canonical(artifact)
        digest = sha256(payload)
        nonce = os.urandom(12)
        path = self.root / (digest + ".enc")
        path.write_bytes(nonce + self.aead.encrypt(nonce, payload, digest.encode()))
        return digest

    def read(self, digest: str) -> dict[str, Any]:
        content = (self.root / (digest + ".enc")).read_bytes()
        decoded = self.aead.decrypt(content[:12], content[12:], digest.encode())
        if sha256(decoded) != digest:
            raise ValueError("Stored artifact checksum mismatch")
        return json.loads(decoded)
