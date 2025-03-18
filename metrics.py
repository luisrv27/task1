import torch as pt

def F1Score(y_true:pt.Tensor, y_pred:pt.Tensor)->float: 
    TP = ((y_true == 1) & (y_pred == 1)).sum().item()
    FP = ((y_true == 0) & (y_pred == 1)).sum().item() 
    FN = ((y_true == 1) & (y_pred == 0)).sum().item() 
    #TN = ((y_true == 0) & (y_pred == 0)).sum().item()  
    #print(TP,FP,FN,TN)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

def Accuracy(y_true:pt.Tensor, y_pred:pt.Tensor)->float: 
    return (y_true == y_pred).float().mean().item()