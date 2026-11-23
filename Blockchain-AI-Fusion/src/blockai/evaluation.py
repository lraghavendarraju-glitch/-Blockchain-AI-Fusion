from __future__ import annotations
from collections import Counter,defaultdict
from typing import Sequence
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
                             confusion_matrix,roc_auc_score,average_precision_score)
from scipy import stats
from .schemas import Workflow, ATTACKS
from .provenance import step_evidence


def _split(items: list[Workflow], test_size: float, seed: int) -> tuple[list[Workflow],list[Workflow]]:
    if len(items)<3:
        raise ValueError("Need >=3 workflows to split")
    ids=np.arange(len(items))
    strata=["normal" if not x.label else x.attack_types[0] if x.attack_types else "attack" for x in items]
    counts=Counter(strata)
    stratify=strata if min(counts.values())>=2 and round(len(items)*test_size)>=len(counts) else None
    if stratify is None:
        binary=[x.label for x in items]
        counts=Counter(binary)
        stratify=binary if min(counts.values())>=2 and round(len(items)*test_size)>=len(counts) else None
    a,b=train_test_split(ids,test_size=test_size,random_state=seed,stratify=stratify)
    return [items[i] for i in a],[items[i] for i in b]


def split_workflows(items: list[Workflow], seed: int=42, protocol: str="random",
                    held_out_attack: str | None = None) -> tuple[list[Workflow],list[Workflow],list[Workflow]]:
    """Main P1: 60/10/30. P2: template-disjoint. P3: leave-one-attack-out."""
    if protocol == "random":
        train_val,test=_split(items,.30,seed)
    elif protocol == "template":
        rng=np.random.default_rng(seed)
        all_templates=sorted({x.template_id for x in items})
        if len(all_templates)<4:
            raise ValueError("Need >=4 templates for template holdout")
        unseen=set(rng.choice(all_templates,size=max(1,round(.3*len(all_templates))),replace=False))
        train_val=[x for x in items if x.template_id not in unseen]
        test=[x for x in items if x.template_id in unseen]
    elif protocol == "attack":
        if held_out_attack not in ATTACKS:
            raise ValueError("Pass --held-out-attack to use attack protocol")
        normal=[w for w in items if w.label==0]
        holdout=[w for w in items if held_out_attack in w.attack_types]
        eligible=[w for w in items if w.label and held_out_attack not in w.attack_types]
        if not holdout:
            raise ValueError("No traces of the held-out attack")
        normal_train,normal_test=_split(normal,.3,seed)
        train_val=normal_train+eligible
        test=normal_test+holdout
    else:
        raise ValueError("Unknown protocol: " + protocol)
    # Split 10% total out of the 70% remaining. For template/attack protocols,
    # 1/7 ensures approximately a 60/10/30 total split.
    train,val=_split(train_val,1/7,seed+1)
    rng=np.random.default_rng(seed)
    rng.shuffle(train);rng.shuffle(val);rng.shuffle(test)
    train_ids={x.workflow_id for x in train}
    val_ids={x.workflow_id for x in val}
    test_ids={x.workflow_id for x in test}
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise AssertionError("Data leakage: workflow in multiple splits")
    if protocol=="template":
        if {x.template_id for x in test} & {x.template_id for x in train+val}:
            raise AssertionError("Template leakage")
    if protocol=="attack":
        if any(held_out_attack in x.attack_types for x in train+val):
            raise AssertionError("Held-out attack leaked to training")
    return train,val,test


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                           probabilities: Sequence[float] | None = None) -> dict:
    yt=np.asarray(y_true,dtype=int); yp=np.asarray(y_pred,dtype=int)
    if not len(yt):
        raise ValueError("Empty evaluation set")
    tn,fp,fn,tp=confusion_matrix(yt,yp,labels=[0,1]).ravel()
    result={"n":int(len(yt)),"accuracy":float(accuracy_score(yt,yp)),
            "precision":float(precision_score(yt,yp,zero_division=0)),
            "recall":float(recall_score(yt,yp,zero_division=0)),
            "f1":float(f1_score(yt,yp,zero_division=0)),
            "far":float(fp/max(1,tn+fp)),"tn":int(tn),"fp":int(fp),
            "fn":int(fn),"tp":int(tp)}
    if probabilities is not None and len(np.unique(yt))>1:
        pp=np.asarray(probabilities,dtype=float)
        result["pr_auc"]=float(average_precision_score(yt,pp))
        result["roc_auc"]=float(roc_auc_score(yt,pp))
    return result


def report_metrics(test: Sequence[Workflow], predictions: list[dict]) -> dict:
    lookup={p["workflow_id"]:p for p in predictions}
    truth=[wf.label for wf in test]
    predicted=[int(lookup[wf.workflow_id]["predicted"]) for wf in test]
    prob=[float(lookup[wf.workflow_id]["anomaly_probability"]) for wf in test]
    metrics=classification_metrics(truth,predicted,prob)
    per_attack={}
    normals=[wf for wf in test if not wf.label]
    for att in ATTACKS:
        subset=normals+[w for w in test if att in w.attack_types]
        if len(subset)>len(normals):
            per_attack[att]=classification_metrics([w.label for w in subset],
                [int(lookup[w.workflow_id]["predicted"]) for w in subset])
    all_true=[];all_pred=[]
    exact=[]
    for wf in test:
        if not wf.label: continue
        step_probs=lookup[wf.workflow_id]["step_probabilities"]
        labels=set(wf.attacked_step_ids)
        predicted_steps={sid for sid,p in step_probs.items() if p>=.5}
        exact.append(predicted_steps==labels)
        for sid,p in step_probs.items():
            all_true.append(int(sid in labels));all_pred.append(int(p>=.5))
    metrics["step_f1"]=float(f1_score(all_true,all_pred,zero_division=0)) if all_true else None
    metrics["step_exact_match_accuracy"]=float(np.mean(exact)) if exact else None
    return {"overall":metrics,"by_attack":per_attack}


def bootstrap_confidence_intervals(test: Sequence[Workflow], predictions: list[dict],
                                   reps: int=1000, seed: int=123) -> dict:
    lookup={p["workflow_id"]:p for p in predictions}
    ids=np.arange(len(test));rng=np.random.default_rng(seed)
    observations={"accuracy":[],"precision":[],"recall":[],"f1":[],"far":[]}
    for _ in range(reps):
        chosen=rng.choice(ids,size=len(ids),replace=True)
        y=[test[i].label for i in chosen]
        p=[lookup[test[i].workflow_id]["predicted"] for i in chosen]
        vals=classification_metrics(y,p)
        for k in observations: observations[k].append(vals[k])
    return {k:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
            for k,v in observations.items()}


def compare_paired_runs(methods: dict[str,list[float]], proposed: str="ProvSeqGAT") -> dict:
    base=np.asarray(methods[proposed],dtype=float)
    output={}
    for name,values in methods.items():
        if name==proposed: continue
        b=np.asarray(values,dtype=float)
        if len(b)!=len(base): raise ValueError("Paired methods require identical run count")
        d=base-b;n=len(d)
        if n<3:
            continue
        sd=float(np.std(d,ddof=1)); mean=float(np.mean(d))
        t=float(stats.ttest_rel(base,b).pvalue) if sd>0 else 1.
        try: w=float(stats.wilcoxon(d).pvalue) if np.any(d!=0) else 1.
        except ValueError: w=1.
        shapiro=float(stats.shapiro(d).pvalue) if 3<=n<=5000 and sd>0 else None
        lo,hi=stats.t.interval(.95,n-1,loc=mean,scale=sd/np.sqrt(n)) if sd>0 else (mean,mean)
        output[name]={"mean_difference":mean,"95ci":[float(lo),float(hi)],
                      "cohens_dz":mean/sd if sd>0 else None,
                      "hedges_g_paired":(mean/sd)*(1-3/(4*(n-1)-1)) if sd>0 and n>2 else None,
                      "paired_t_p":t,"wilcoxon_p":w,"shapiro_p":shapiro}
    # Holm correction on the family of paired t-tests.
    ordered=sorted(output,key=lambda x:output[x]["paired_t_p"])
    corrected=[]
    for j,name in enumerate(ordered): corrected.append(min(1.,output[name]["paired_t_p"]*(len(ordered)-j)))
    for j,name in enumerate(ordered): output[name]["holm_adjusted_p"]=max(corrected[:j+1])
    return output


def save_json(path: str | Path, data) -> None:
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,allow_nan=False,default=float),encoding="utf-8")
