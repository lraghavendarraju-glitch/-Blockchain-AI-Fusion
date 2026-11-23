"""Source-aware adapters for WfFormat and OpenML experiment metadata.

No native attack labels or artifact bytes are asserted for either corpus.
Every inferred lineage field is explicitly identified as such.
"""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import quote
import requests
import networkx as nx
from .schemas import Workflow,Step,ROLE_POLICY,STEP_TYPES
from .provenance import sha256,parent_commitment,report_digest


def parse_timestamp(value, default: float) -> float:
    if not value: return default
    if isinstance(value,(int,float)): return float(value)
    try: return datetime.fromisoformat(str(value).replace("Z","+00:00")).timestamp()
    except (ValueError,TypeError): return default


def _category(name: str) -> str:
    name=name.lower()
    if any(v in name for v in ("ingest", "read", "load", "download", "split")): return "ingest"
    if any(v in name for v in ("valid", "check")): return "validate"
    if any(v in name for v in ("preprocess", "clean", "filter", "normalize")): return "preprocess"
    if any(v in name for v in ("feature", "transform", "project", "align")): return "feature_engineer"
    if any(v in name for v in ("train", "fit", "learn")): return "train"
    if any(v in name for v in ("tune", "optimiz")): return "tune"
    if any(v in name for v in ("evaluate", "score", "test")): return "evaluate"
    if "register" in name: return "register"
    if "deploy" in name: return "deploy"
    if any(v in name for v in ("monitor", "track")): return "monitor"
    if "audit" in name: return "audit"
    return "feature_engineer"  # semantic mapping, not an observed task label


def _wf_from_description(wf_id: str, observed_tasks: list[dict], source: str,
                         meta: dict) -> Workflow:
    if not observed_tasks: raise ValueError("No observed tasks")
    graph=nx.DiGraph()
    for t in observed_tasks: graph.add_node(str(t["id"]))
    for t in observed_tasks:
        for parent in t.get("parents",[]):
            if str(parent) not in graph:
                raise ValueError(f"Unknown dependency {parent}")
            graph.add_edge(str(parent),str(t["id"]))
    if not nx.is_directed_acyclic_graph(graph): raise ValueError("Real workflow contains dependency cycle")
    by_id={str(t["id"]):t for t in observed_tasks}
    topological=list(nx.lexicographical_topological_sort(graph))
    output_hashes={}
    steps=[]
    for i,original_id in enumerate(topological):
        task=by_id[original_id]
        sid=f"{wf_id}:S{i:05d}"
        parents=[f"{wf_id}:S{topological.index(str(p)):05d}" for p in graph.predecessors(original_id)]
        deps=[(p,output_hashes[p]) for p in parents]
        input_hash=parent_commitment(deps)
        step_type=_category(str(task.get("name", "")))
        artifact={"source":source,"task_id":original_id,"observed_name":task.get("name"),
                  "file_refs":task.get("files",[]),"command":task.get("command",{})}
        output_hash=sha256(artifact)
        metric=task.get("metric")
        try: metric=float(metric) if metric is not None else None
        except (TypeError,ValueError): metric=None
        role="administrator"  # simulated governance role, NOT observed certification
        timestamp=parse_timestamp(task.get("timestamp"),1720000000.+i*30)
        step=Step(sid,step_type,str(meta.get("actor","external")),role,parents,timestamp,
                  max(0.,float(task.get("duration") or 0.)),
                  {"learning_rate":.001,"feature_count":0,"batch_size":32},
                  metric,artifact,input_hash,output_hash,"",sha256({"model":wf_id}),1,i)
        step.params_salt=sha256({"wf":wf_id,"task":original_id})[:32]
        step.report_hash=report_digest(step)
        steps.append(step)
        output_hashes[sid]=output_hash
    return Workflow(wf_id,template_id=int(int(sha256(source)[:8],16)%20),version=1,steps=steps,
                    expected_step_ids=[s.step_id for s in steps],provenance_source=source,
                    metadata={**meta,"artifact_hash_scope":"metadata_only_without_artifact_bytes",
                              "role_source":"synthetic_test_mapping_not_observed",
                              "lineage_source":"observed" if source=="wfcommons" else "inferred"})


def parse_wfcommons_file(path: str | Path) -> Workflow:
    """Supports WfFormat 1.6 specification/execution and older flat task layouts."""
    path=Path(path)
    data=json.loads(path.read_text(encoding="utf-8"))
    workflow=data.get("workflow",{})
    spec=workflow.get("specification",{})
    specification=spec.get("tasks") or workflow.get("tasks") or data.get("tasks")
    if not specification: raise ValueError(f"No WfFormat tasks in {path}")
    executions={str(t.get("id")):t for t in workflow.get("execution",{}).get("tasks",[])}
    observed=[]
    for t in specification:
        task_id=str(t.get("id",t.get("name","")))
        if not task_id: raise ValueError("Task missing id")
        execution=executions.get(task_id,{})
        observed.append({"id":task_id,"parents":t.get("parents",[]),
                         "name":t.get("name",task_id),
                         "timestamp":execution.get("executedAt"),
                         "duration":execution.get("runtimeInSeconds",0),
                         "command":execution.get("command",{}),
                         "files":{"input":t.get("inputFiles",[]),"output":t.get("outputFiles",[])}})
    wf_id="WFC_"+sha256({"source":str(path),"name":data.get("name")})[:12]
    return _wf_from_description(wf_id,observed,"wfcommons",
                               {"source_name":data.get("name",path.stem),"source_file":str(path),
                                "runtime_system":data.get("runtimeSystem",{}),"actor":"PUBLIC_WFC_USER"})


def parse_openml_run(run: dict) -> Workflow:
    """OpenML run metadata -> explicitly INFERRED seven-stage workflow.

    OpenML does not generally expose the full execution DAG or artifact bytes.
    This is valid for *inferred-workflow* experiments only.
    """
    rid=str(run.get("run_id",run.get("id","unknown")))
    task=run.get("task_id")
    flow=run.get("flow_id")
    uploader=run.get("uploader", "unknown")
    params=run.get("parameter_settings",{})
    stages=["ingest","validate","preprocess","feature_engineer","train","evaluate","audit"]
    epoch=parse_timestamp(run.get("upload_time"),1720000000.)
    observed=[]
    for i,stage in enumerate(stages):
        observed.append({"id":f"STAGE_{i:02d}","parents":[] if i==0 else [f"STAGE_{i-1:02d}"],
                         "name":stage,"timestamp":epoch+i*30,"duration":0.,
                         "files":[],"command":{"inferred_from_run":rid,"task_id":task,"flow_id":flow,
                                                "parameters":params if stage in ("train","feature_engineer") else {}}})
    return _wf_from_description(f"OPENML_{rid}",observed,"openml_inferred",
                                {"run_id":rid,"task_id":task,"flow_id":flow,
                                 "actor":f"OPENML_USER_{uploader}","inferred_stages":stages})


def fetch_wfcommons(limit: int, destination: str | Path,
                    applications: tuple[str,...] = ("pegasus/","makeflow/"),
                    github_token: str | None = None) -> list[dict]:
    """Downloads up to `limit` JSON WfInstances and saves a SHA-256 manifest."""
    if limit<1: raise ValueError("Limit must be positive")
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    headers={"Accept":"application/vnd.github+json","User-Agent":"blockai-research"}
    if github_token: headers["Authorization"]="Bearer "+github_token
    tree=requests.get("https://api.github.com/repos/wfcommons/WfInstances/git/trees/main?recursive=1",
                      headers=headers,timeout=60)
    tree.raise_for_status()
    paths=[x["path"] for x in tree.json()["tree"] if x.get("type")=="blob"
           and x["path"].lower().endswith(".json") and x["path"].startswith(applications)]
    selected=sorted(paths)[:limit]
    manifest=[]
    for name in selected:
        url="https://raw.githubusercontent.com/wfcommons/WfInstances/main/"+quote(name,safe="/")
        resp=requests.get(url,timeout=120,headers={"User-Agent":"blockai-research"});resp.raise_for_status()
        path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(resp.content)
        manifest.append({"relative_path":name,"sha256":hashlib.sha256(resp.content).hexdigest(),"url":url})
    (dest/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


def fetch_openml(limit: int, destination: str | Path,
                 task_ids: list[int] | None = None) -> list[dict]:
    try: import openml
    except ImportError as exc: raise RuntimeError("pip install 'blockai-fusion[experiments]' for OpenML") from exc
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    listed=openml.runs.list_runs(size=limit,task=task_ids)
    if hasattr(listed,"to_dict"): records=listed.to_dict(orient="records")
    else: records=list(listed.values())
    manifest=[]
    for row in records[:limit]:
        rid=int(row.get("run_id",row.get("run_id",row.get("id"))))
        run=openml.runs.get_run(rid)
        obj={"run_id":rid,"task_id":getattr(run,"task_id",row.get("task_id")),
             "flow_id":getattr(run,"flow_id",row.get("flow_id")),
             "uploader":row.get("uploader"),"upload_time":str(row.get("upload_time")),
             "parameter_settings":getattr(run,"parameter_settings",{}),
             "evaluations":getattr(run,"evaluations",{})}
        # Keep metadata only. Some entries may require JSON coercion.
        content=json.dumps(obj,default=str,sort_keys=True).encode()
        path=dest/f"run_{rid}.json";path.write_bytes(content)
        manifest.append({"run_id":rid,"relative_path":path.name,"sha256":hashlib.sha256(content).hexdigest()})
    (dest/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


def load_real_workflows(source: str, root: str | Path, limit: int | None = None) -> list[Workflow]:
    root=Path(root)
    if source=="wfcommons":
        paths=sorted(p for p in root.rglob("*.json") if p.name!="manifest.json")
        parser=parse_wfcommons_file
    elif source=="openml":
        paths=sorted(p for p in root.glob("run_*.json"))
        parser=lambda path:parse_openml_run(json.loads(path.read_text()))
    else: raise ValueError("source must be wfcommons or openml")
    return [parser(p) for p in paths[:limit]]
