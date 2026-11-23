import numpy as np
import torch
from blockai.features import FeatureEncoder
from blockai.model import ProvSeqGAT,collate_graphs
from blockai.training import Trainer,TrainConfig,choose_thresholds
from blockai.evaluation import split_workflows,report_metrics,classification_metrics


def test_shapes_autograd(small_dataset):
    enc=FeatureEncoder().fit(small_dataset[:50])
    graphs=[enc.transform(w) for w in small_dataset[50:54]]
    assert all(s.numeric.shape[1]==40 for s in graphs)
    batch=collate_graphs(graphs)
    model=ProvSeqGAT()
    output=model(batch)
    assert output['workflow_logits'].shape==(4,)
    assert output['step_logits'].shape==(sum(len(w.steps) for w in small_dataset[50:54]),)
    output['workflow_logits'].sum().backward()
    assert model.project[0].weight.grad is not None
    assert model.n_parameters>100000


def test_training_reload_and_test_prediction(small_dataset,tmp_path):
    train,val,test=split_workflows(small_dataset,42)
    trainer=Trainer(TrainConfig(epochs=2,patience=2,batch_size=8,seed=101,warmup_epochs=1))
    result=trainer.train(train,val)
    assert result['epochs_run']==2
    assert 0<=result['thresholds']['theta1']<=1
    before=trainer.predict(test[:3])
    trainer.save(tmp_path)
    recovered=Trainer.load(tmp_path)
    after=recovered.predict(test[:3])
    for a,b in zip(before,after):
        assert a['workflow_id']==b['workflow_id']
        assert a['anomaly_probability']==b['anomaly_probability']
    metrics=report_metrics(test,trainer.predict(test))
    assert 0<=metrics['overall']['accuracy']<=1


def test_holdout_splits(small_dataset):
    tr,val,te=split_workflows(small_dataset,42,'template')
    assert not ({x.template_id for x in tr+val}&{x.template_id for x in te})
    tr,val,te=split_workflows(small_dataset,42,'attack',held_out_attack='model_replacement')
    assert all('model_replacement' not in w.attack_types for w in tr+val)
    assert any('model_replacement' in w.attack_types for w in te)


def test_threshold_validation_no_positive_class():
    result=choose_thresholds([0,0,0],[.1,.2,.3])
    assert result['theta2_support_sufficient'] is False
