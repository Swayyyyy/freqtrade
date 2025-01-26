#计算auc
from sklearn.metrics import roc_auc_score
import numpy as np

def auc_score(y_true, y_pred, **kwargs):
    return roc_auc_score(y_true, y_pred)

#多分类auc
def auc_score_for_multiclass(y_true, y_pred, multi_class="ovo", **kwargs):
    return roc_auc_score(y_true, y_pred, multi_class=multi_class, **kwargs)

