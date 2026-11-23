import json
from pathlib import Path
from blockai.real_datasets import parse_wfcommons_file,parse_openml_run
from blockai.provenance import step_evidence


def test_wfformat_parser(tmp_path):
    sample={"name":"two-step-test","schemaVersion":"1.6","workflow":{
        "specification":{"tasks":[{"id":"A","name":"load","parents":[],"outputFiles":["a.csv"]},
                                  {"id":"B","name":"train","parents":["A"],"inputFiles":["a.csv"]}]},
        "execution":{"tasks":[{"id":"A","runtimeInSeconds":1.2,"executedAt":"2024-01-01T00:00:00Z"},
                              {"id":"B","runtimeInSeconds":4.,"executedAt":"2024-01-01T00:00:01Z"}]}}}
    p=tmp_path/'test.json';p.write_text(json.dumps(sample))
    wf=parse_wfcommons_file(p)
    assert len(wf.steps)==2
    assert len(wf.steps[1].parents)==1
    assert wf.metadata['artifact_hash_scope']=='metadata_only_without_artifact_bytes'
    assert all(all(step_evidence(s,wf)[k] for k in ('seq','acc','int')) for s in wf.steps)


def test_openml_is_marked_inferred():
    wf=parse_openml_run({'run_id':442,'task_id':3,'flow_id':5,'uploader':19})
    assert wf.provenance_source=='openml_inferred'
    assert wf.metadata['lineage_source']=='inferred'
    assert len(wf.steps)==7
