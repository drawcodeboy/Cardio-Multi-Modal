import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

def get_metrics(outputs, targets, class_num:int=7):
    results = {
        "Accuracy": 0.,
        "F1-Score(Micro)":0.,
        #"Precision(Micro)":0.,
        #"Recall(Micro)":0.,
        "F1-Score(Macro)":0.,
        "Precision(Macro)":0.,
        "Recall(Macro)":0.,
    }
    
    accuracy = np.mean(outputs == targets)
    
    micro_f1 = f1_score(targets, outputs, average='micro')
    macro_f1 = f1_score(targets, outputs, average='macro')
    precision = precision_score(targets, outputs, average='macro')
    recall = recall_score(targets, outputs, average='macro')
    
    results['Accuracy'] = accuracy
    results["F1-Score(Micro)"] = micro_f1
    results["F1-Score(Macro)"] = macro_f1
    results["Precision(Macro)"] = precision
    results["Recall(Macro)"] = recall
    
    '''
    result_matrix = np.zeros((class_num, 4)) # [cls0: [TP, FP, TN, FN], cls1: [TP, FP, TN, FN], ...]
    
    for cls in range(class_num):
        # cls를 Positive 취급
        binary_targets = (targets == cls).astype(int)
        binary_outputs = (outputs == cls).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(binary_targets, binary_outputs, labels=[0, 1]).ravel()
        
        result_matrix[cls] = np.array([tp, fp, tn, fn])
    
    micro_infos = result_matrix.sum(axis=0)
    
    results['Accuracy'] = (micro_infos[0]+micro_infos[2])/micro_infos.sum()
    results['Precision(Micro)'] = (micro_infos[0])/(micro_infos[0]+micro_infos[1])
    results['Recall(Micro)'] = (micro_infos[0])/(micro_infos[0]+micro_infos[3])
    results['F1-Score(Micro)'] = (2*results['Precision(Micro)']*results['Recall(Micro)'])/(results['Precision(Micro)']+results['Recall(Micro)'])
    
    f1_macro, pre_macro, rec_macro = [], [], []
    for cls_res in result_matrix:
        precision_temp = (cls_res[0])/(1e-7 if (cls_res[0]+cls_res[1]) == 0 else (cls_res[0]+cls_res[1]))
        recall_temp = (cls_res[0])/(1e-7 if (cls_res[0]+cls_res[3]) == 0 else (cls_res[0]+cls_res[3]))
        pre_macro.append(precision_temp)
        rec_macro.append(recall_temp)
        f1_temp = (2*precision_temp*recall_temp)/(1e-7 if (precision_temp+recall_temp) == 0 else (precision_temp+recall_temp)) 
        f1_macro.append(f1_temp)
        
    results['Precision(Macro)'] = sum(pre_macro)/class_num
    results['Recall(Macro)'] = sum(rec_macro)/class_num                     
    results['F1-Score(Macro)'] = sum(f1_macro)/class_num
    '''
    
    return results