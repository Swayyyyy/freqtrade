#根据给定的标签，只计算其中给定标签的复合准确率,即将这几类数据复合起来计算准确率
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def accuracy_score_for_labels(y_true, y_pred, labels, le, **kwargs):
    y_true_label_id = le.transform(y_true)
    data = []
    for label in labels:
        y_true_label = y_true_label_id[y_true == label]
        if len(y_true_label) == 0:
            continue
        y_pred_label = y_pred[y_true == label]
        data.append(accuracy_score(y_true_label, y_pred_label))
    if len(data) == 0:
        return 0
    return np.mean(data)

def accuracy_score_for_labels_with_weight(y_true, y_pred, labels, weights, le, **kwargs):
    #根据给定的标签，只计算其中给定标签的复合准确率,即将这几类数据复合起来计算准确率
    #weights是每个标签的权重
    #返回复合准确率         
    y_true_label_id = le.transform(y_true)
    data = []
    for label in labels:
        y_true_label = y_true_label_id[y_true == label]
        y_pred_label = y_pred[y_true == label]
        data.append(accuracy_score(y_true_label, y_pred_label) * weights[label])
    return np.mean(data)

