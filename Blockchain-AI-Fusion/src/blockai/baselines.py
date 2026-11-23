from __future__ import annotations
from typing import Sequence
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.ensemble import IsolationForest,HistGradientBoostingClassifier
from sklearn.svm import OneClassSVM
from scipy.special import expit
from .features import FeatureEncoder, GraphSample
from .schemas import Workflow,STEP_TYPES
from .provenance import step_evidence
from .training import choose_thresholds,set_seed
from .evaluation import classification_metrics


def aggregate_features(sample: GraphSample) -> np.ndarray:
    x=sample.numeric
    return np.concatenate([x.mean(0),x.std(0),x.max(0),sample.global_features])


def graph_structural_features(sample: GraphSample) -> np.ndarray:
    n=len(sample.numeric)
    source,target=sample.edge_index
    indegree=np.bincount(target,minlength=n)
    outdegree=np.bincount(source,minlength=n)
    types=np.bincount(sample.type_ids,minlength=13)[1:13]/max(1,n)
    return np.concatenate([np.histogram(indegree,bins=[0,1,2,3,5,1e9])[0]/n,
                           np.histogram(outdegree,bins=[0,1,2,3,5,1e9])[0]/n,
                           types,np.array([n, len(source)/n])])


def _scaled_scores(train: np.ndarray, validation: np.ndarray, test: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    mu=float(np.mean(train));sigma=max(1e-6,float(np.std(train)))
    return expit((validation-mu)/sigma),expit((test-mu)/sigma)


class _LSTMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(40,32,batch_first=True)
        self.reconstruct=nn.Linear(32,40)
    def forward(self,x):
        y,_=self.lstm(x)
        return self.reconstruct(y)


class _DeepLog(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(13,16)
        self.gru=nn.GRU(16,32,batch_first=True)
        self.head=nn.Linear(32,13)
    def forward(self,x):
        y,_=self.gru(self.embedding(x))
        return self.head(y)


def _sequence_model_scores(train: list[GraphSample], val: list[GraphSample],
                           test: list[GraphSample], method: str, seed: int,
                           epochs: int = 5) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    set_seed(seed)
    model=_LSTMAE() if method=="LSTM_Autoencoder" else _DeepLog()
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    normal=[s for s in train if s.label==0]
    if not normal: normal=train
    for _ in range(epochs):
        np.random.shuffle(normal)
        model.train()
        for i in range(0,len(normal),32):
            group=normal[i:i+32]
            if method=="LSTM_Autoencoder":
                arr=[torch.tensor(s.numeric,dtype=torch.float32) for s in group]
                padded=pad_sequence(arr,batch_first=True)
                mask=torch.arange(padded.size(1))[None,:] < torch.tensor([len(a) for a in arr])[:,None]
                decoded=model(padded)
                loss=((decoded-padded).square().mean(-1)*mask).sum()/mask.sum().clamp_min(1)
            else:
                arr=[torch.tensor(s.type_ids,dtype=torch.long) for s in group]
                inp=pad_sequence([a[:-1] for a in arr],batch_first=True)
                target=pad_sequence([a[1:] for a in arr],batch_first=True)
                logits=model(inp)
                loss=nn.functional.cross_entropy(logits.reshape(-1,13),target.reshape(-1),ignore_index=0)
            optimizer.zero_grad();loss.backward();optimizer.step()
    def score(dataset):
        model.eval(); scores=[]
        with torch.no_grad():
            for s in dataset:
                if method=="LSTM_Autoencoder":
                    x=torch.tensor(s.numeric,dtype=torch.float32).unsqueeze(0)
                    scores.append(float((model(x)-x).square().mean()))
                else:
                    ids=torch.tensor(s.type_ids,dtype=torch.long)
                    if len(ids)<2: scores.append(0.);continue
                    logits=model(ids[:-1][None,:])[0]
                    scores.append(float(nn.functional.cross_entropy(logits,ids[1:])))
        return np.asarray(scores)
    return score(normal),score(val),score(test)


def run_baselines(train: Sequence[Workflow], val: Sequence[Workflow], test: Sequence[Workflow],
                  encoder: FeatureEncoder, seed: int=42, quick: bool=False) -> dict:
    """Returns metrics and per-workflow predictions for each baseline.

    'ProvenanceGraph_IF_proxy' is a transparent lightweight graph-structural
    substitute for comparison; it MUST NOT be reported as original UNICORN.
    """
    train=[encoder.transform(w) for w in train]
    validation=[encoder.transform(w) for w in val]
    tests=[encoder.transform(w) for w in test]
    yval=np.array([int(s.label) for s in validation]); ytest=np.array([int(s.label) for s in tests])
    normal_idx=[i for i,s in enumerate(train) if s.label==0]
    agg_train=np.stack([aggregate_features(s) for s in train]);
    agg_val=np.stack([aggregate_features(s) for s in validation]);
    agg_test=np.stack([aggregate_features(s) for s in tests]);
    result={}

    def record(name: str, val_scores: np.ndarray, test_scores: np.ndarray,
               fixed_threshold: float | None = None):
        theta=float(choose_thresholds(yval,val_scores)["theta1"]) if fixed_threshold is None else fixed_threshold
        pred=(test_scores>=theta).astype(int)
        result[name]={"metrics":classification_metrics(ytest,pred,test_scores),"threshold":theta,
                      "predictions":{s.workflow_id:{"probability":float(p),"predicted":int(label)}
                                     for s,p,label in zip(tests,test_scores,pred)}}

    # Rule-only checks are pre-specified, not learned.
    # Rule-only detection is derived from independent, reversible validation indicators.
    # Reconstruct deterministic rule results from the integrity dimensions of raw sample features
    # (standardized features invert using the training scaler in a model-independent way).
    def rule_score(samples):
        out=[]
        for s in samples:
            orig=encoder.numeric_scaler.inverse_transform(s.numeric)
            out.append(float(np.any(np.min(orig[:,24:27],axis=1)<.5)))
        return np.asarray(out)
    record("RuleOnly",rule_score(validation),rule_score(tests),fixed_threshold=.5)
    for name,estimator in [
        ("IsolationForest",IsolationForest(n_estimators=60 if quick else 150,random_state=seed,contamination="auto")),
        ("OneClassSVM",OneClassSVM(kernel="rbf",nu=.15,gamma="scale"))]:
        estimator.fit(agg_train[normal_idx])
        norm=-estimator.score_samples(agg_train[normal_idx])
        v=-estimator.score_samples(agg_val);t=-estimator.score_samples(agg_test)
        vs,ts=_scaled_scores(norm,v,t)
        record(name,vs,ts)
    try:
        from xgboost import XGBClassifier
        supervised=XGBClassifier(n_estimators=60 if quick else 150,max_depth=4,learning_rate=.05,
                                 subsample=.9,colsample_bytree=.9,n_jobs=2,random_state=seed,
                                 eval_metric="logloss")
    except ImportError:
        supervised=HistGradientBoostingClassifier(max_iter=60 if quick else 150,random_state=seed)
    supervised.fit(agg_train,np.array([s.label for s in train]))
    record("XGBoost" if type(supervised).__name__=="XGBClassifier" else "HistGradientBoosting_fallback",
           supervised.predict_proba(agg_val)[:,1],supervised.predict_proba(agg_test)[:,1])

    structural=[np.stack([graph_structural_features(s) for s in group]) for group in (train,validation,tests)]
    graph_if=IsolationForest(n_estimators=60 if quick else 150,random_state=seed)
    graph_if.fit(structural[0][normal_idx])
    norm=-graph_if.score_samples(structural[0][normal_idx]);v=-graph_if.score_samples(structural[1]);t=-graph_if.score_samples(structural[2])
    vs,ts=_scaled_scores(norm,v,t)
    record("ProvenanceGraph_IF_proxy",vs,ts)
    for method in ("LSTM_Autoencoder","DeepLog"):
        norm,v,t=_sequence_model_scores(train,validation,tests,method,seed,epochs=2 if quick else 8)
        vs,ts=_scaled_scores(norm,v,t)
        record(method,vs,ts)
    return result
