"""Leakage-controlled supervised training, threshold selection and prediction."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import copy
import json
import math
from pathlib import Path
import random
from typing import Sequence
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_curve, precision_recall_curve
from torch.utils.data import DataLoader
from .schemas import Workflow
from .features import FeatureEncoder, GraphSample
from .model import ProvSeqGAT, collate_graphs


@dataclass
class TrainConfig:
    epochs: int = 100
    patience: int = 10
    batch_size: int = 64
    learning_rate: float = .001
    weight_decay: float = .0001
    dropout: float = .3
    gru_hidden: int = 64
    gat_heads: int = 4
    gat_layers: int = 2
    warmup_epochs: int = 5
    seed: int = 42
    device: str = "auto"
    use_gru: bool = True
    use_integrity: bool = True
    use_actor: bool = True
    use_temporal: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_thresholds(y_true: Sequence[int], y_prob: Sequence[float],
                      target_precision: float = .99) -> dict[str, float | bool]:
    labels=np.asarray(y_true,dtype=int); proba=np.asarray(y_prob,dtype=float)
    if len(np.unique(labels)) < 2:
        return {"theta1": .5, "theta2": 1.0, "theta2_support_sufficient": False}
    fpr,tpr,thr=roc_curve(labels,proba)
    j=tpr-fpr
    finite=np.where(np.isfinite(thr))[0]
    idx=int(finite[np.argmax(j[finite])]) if len(finite) else 0
    theta1=float(np.clip(thr[idx],0,1))
    # The paper's theta2 is only achievable if validation support exists.
    candidates=np.unique(np.clip(proba,0,1))
    viable=[]
    for theta in candidates:
        chosen=proba>=theta
        if chosen.sum() >= 1 and float(labels[chosen].mean()) >= target_precision:
            viable.append(float(theta))
    theta2=max(theta1,min(viable)) if viable else 1.0
    return {"theta1": theta1,"theta2": theta2,
            "theta2_support_sufficient": bool(viable)}


class Trainer:
    def __init__(self, config: TrainConfig):
        set_seed(config.seed)
        self.config = config
        self.device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else
                                   "cpu" if config.device == "auto" else config.device)
        self.encoder=FeatureEncoder()
        self.model=ProvSeqGAT(config.gru_hidden, config.gat_heads, config.gat_layers, config.dropout,
                               use_gru=config.use_gru, use_integrity=config.use_integrity,
                               use_actor=config.use_actor, use_temporal=config.use_temporal).to(self.device)
        self.thresholds={"theta1":.5,"theta2":.8,"theta2_support_sufficient":False}
        self.history=[]

    def _loader(self, data: Sequence[GraphSample], shuffle: bool) -> DataLoader:
        return DataLoader(list(data),batch_size=self.config.batch_size,shuffle=shuffle,
                          collate_fn=collate_graphs,num_workers=0)

    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader) -> tuple[np.ndarray,np.ndarray]:
        self.model.eval()
        labels=[]; probs=[]
        for batch in loader:
            batch=batch.to(self.device)
            output=self.model(batch)
            labels.extend(batch.labels.cpu().numpy().tolist())
            probs.extend(torch.sigmoid(output["workflow_logits"]).cpu().numpy().tolist())
        return np.asarray(labels),np.asarray(probs)

    def train(self, train: Sequence[Workflow], validation: Sequence[Workflow]) -> dict:
        set_seed(self.config.seed)
        self.encoder.fit(train)
        train_x=[self.encoder.transform(x) for x in train]
        val_x=[self.encoder.transform(x) for x in validation]
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=self.config.learning_rate,
                                    weight_decay=self.config.weight_decay)
        steps=max(1,self.config.epochs)
        def multiplier(epoch):
            if epoch < self.config.warmup_epochs:
                return max(.1,(epoch+1)/max(1,self.config.warmup_epochs))
            fraction=(epoch-self.config.warmup_epochs)/max(1,steps-self.config.warmup_epochs)
            return max(.01,.5*(1+math.cos(math.pi*fraction)))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)
        positive_loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.33,device=self.device))
        step_loss=nn.BCEWithLogitsLoss()
        best=-float("inf");best_state=None;wait=0
        for epoch in range(self.config.epochs):
            self.model.train()
            train_losses=[]
            for batch in self._loader(train_x,shuffle=True):
                batch=batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits=self.model(batch)
                loss=positive_loss(logits["workflow_logits"],batch.labels) + .3*step_loss(logits["step_logits"],batch.step_labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            scheduler.step()
            targets,prob=self._evaluate_loader(self._loader(val_x,shuffle=False))
            auc=float(average_precision_score(targets,prob)) if len(set(targets)) > 1 else 0.
            self.history.append({"epoch":epoch+1,"loss":float(np.mean(train_losses)),"validation_pr_auc":auc})
            if auc>best+1e-6:
                best=auc; best_state=copy.deepcopy(self.model.state_dict()); wait=0
            else:
                wait+=1
                if wait>=self.config.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        truth,prob=self._evaluate_loader(self._loader(val_x,shuffle=False))
        self.thresholds=choose_thresholds(truth,prob)
        return {"best_val_pr_auc":best,"epochs_run":len(self.history),"thresholds":self.thresholds,
                "model_parameters":self.model.n_parameters,"device":str(self.device)}

    @torch.no_grad()
    def predict(self, workflows: Sequence[Workflow]) -> list[dict]:
        self.model.eval()
        samples=[self.encoder.transform(wf) for wf in workflows]
        predictions=[]
        for batch in self._loader(samples,shuffle=False):
            batch=batch.to(self.device)
            pred=self.model(batch)
            wf_prob=torch.sigmoid(pred["workflow_logits"]).cpu().numpy()
            step_prob=torch.sigmoid(pred["step_logits"]).cpu().numpy()
            start=0
            for i,n in enumerate(batch.lengths):
                p=float(wf_prob[i]); theta1=float(self.thresholds["theta1"]); theta2=float(self.thresholds["theta2"])
                risk="normal" if p<theta1 else "suspicious" if p<theta2 else "compromised"
                predictions.append({"workflow_id":batch.workflow_ids[i],"label":int(batch.labels[i].item()),
                                    "anomaly_probability":p,"predicted":int(p>=theta1),"risk":risk,
                                    "step_probabilities":dict(zip(batch.step_ids[i], [float(z) for z in step_prob[start:start+n]]))})
                start+=n
        return predictions

    def save(self, root: str | Path) -> None:
        root=Path(root); root.mkdir(parents=True,exist_ok=True)
        torch.save({"state_dict":self.model.state_dict(),"config":asdict(self.config),
                    "thresholds":self.thresholds},root/"provseqgat.pt")
        self.encoder.save(str(root/"encoder.pkl"))
        (root/"history.json").write_text(json.dumps(self.history,indent=2))

    @classmethod
    def load(cls, root: str | Path, device: str = "auto") -> "Trainer":
        root=Path(root)
        try:
            packed=torch.load(root/"provseqgat.pt",map_location="cpu",weights_only=True)
        except TypeError:
            packed=torch.load(root/"provseqgat.pt",map_location="cpu")
        cfg=TrainConfig(**packed["config"]);cfg.device=device
        obj=cls(cfg)
        obj.model.load_state_dict(packed["state_dict"])
        obj.encoder=FeatureEncoder.load(str(root/"encoder.pkl"))
        obj.thresholds=packed["thresholds"]
        return obj
