import numpy as np
from blockai.evaluation import classification_metrics,compare_paired_runs


def test_confusion_counts_and_holm():
    m=classification_metrics([0,0,1,1],[0,1,1,1],[.1,.9,.9,.8])
    assert (m['tn'],m['fp'],m['fn'],m['tp'])==(1,1,0,2)
    assert m['far']==.5
    values={'ProvSeqGAT':[.9,.87,.91,.88,.92],
            'baseline1':[.8,.81,.79,.82,.80],
            'baseline2':[.85,.82,.86,.81,.84]}
    compared=compare_paired_runs(values)
    assert set(compared)=={'baseline1','baseline2'}
    assert all(0<=v['holm_adjusted_p']<=1 for v in compared.values())
