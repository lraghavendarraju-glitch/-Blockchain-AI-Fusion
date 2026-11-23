"""Manuscript-ready evaluation graphics (600 dpi; axes labelled; no chart titles)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path,dpi=600,bbox_inches="tight")
    plt.close(fig)


def generate_figures(workflows, predictions, metrics: dict, output_dir: str | Path) -> list[str]:
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    lookup={p["workflow_id"]:p for p in predictions}
    y=np.array([int(w.label) for w in workflows])
    prob=np.array([float(lookup[w.workflow_id]["anomaly_probability"]) for w in workflows])
    pred=np.array([int(lookup[w.workflow_id]["predicted"]) for w in workflows])
    files=[]
    if len(np.unique(y))>1:
        precision,recall,_=precision_recall_curve(y,prob)
        fig,ax=plt.subplots(figsize=(5.6,4.3))
        ax.plot(recall,precision,linewidth=2)
        ax.set_xlabel("Recall");ax.set_ylabel("Precision")
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.2)
        path=output_dir/"figure_pr_curve.png";_save(fig,path);files.append(str(path))
        fpr,tpr,_=roc_curve(y,prob)
        fig,ax=plt.subplots(figsize=(5.6,4.3))
        ax.plot(fpr,tpr,linewidth=2)
        ax.plot([0,1],[0,1],"--",alpha=.5)
        ax.set_xlabel("False Positive Rate");ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.2)
        path=output_dir/"figure_roc_curve.png";_save(fig,path);files.append(str(path))
    confusion=np.array([[sum((y==0)&(pred==0)),sum((y==0)&(pred==1))],
                        [sum((y==1)&(pred==0)),sum((y==1)&(pred==1))]])
    fig,ax=plt.subplots(figsize=(4.4,3.9))
    ax.imshow(confusion,cmap="Blues")
    ax.set_xticks([0,1],["Normal","Attack"])
    ax.set_yticks([0,1],["Normal","Attack"])
    ax.set_xlabel("Predicted Class");ax.set_ylabel("True Class")
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(confusion[i,j]),ha="center",va="center",
                    color="white" if confusion[i,j]>confusion.max()/2 else "black")
    path=output_dir/"figure_confusion_matrix.png";_save(fig,path);files.append(str(path))
    cats=list(metrics.get("by_attack",{}))
    if cats:
        values=[metrics["by_attack"][key]["recall"] for key in cats]
        fig,ax=plt.subplots(figsize=(8.5,4.5))
        ax.bar(range(len(cats)),values)
        ax.set_xticks(range(len(cats)),[x.replace("_"," ").title() for x in cats],rotation=25,ha="right")
        ax.set_ylabel("Attack Detection Recall")
        ax.set_ylim(0,1);ax.grid(axis="y",alpha=.2)
        path=output_dir/"figure_per_attack_recall.png";_save(fig,path);files.append(str(path))
    fig,ax=plt.subplots(figsize=(6.0,4.0))
    lengths=np.array([len(w.steps) for w in workflows])
    ax.hist(lengths,bins=min(20,max(4,len(set(lengths)))),edgecolor="white")
    ax.set_xlabel("Workflow Length (Steps)");ax.set_ylabel("Number of Workflows")
    path=output_dir/"figure_workflow_lengths.png";_save(fig,path);files.append(str(path))
    return files
