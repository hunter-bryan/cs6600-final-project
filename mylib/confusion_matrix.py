import numpy as np

def confusion_matrix(targets, predicted):
    """
    A confusion matrix C is such that C[i, j] is equal to the number 
    of observations known to be in i and predicted to be in group j.
    """
    # Convert classes into index
    targets = targets.flatten()
    predicted = predicted.flatten()
    labels = np.unique(np.concatenate([targets, predicted], axis=0))
    np.sort(labels)
    nLabels = len(labels)
    
    label_to_index = { lbl: ndx for ndx, lbl in enumerate(labels) } 
    
    y_true = np.array([label_to_index.get(lbl) for lbl in targets])
    y_pred = np.array([label_to_index.get(lbl) for lbl in predicted])
    
    cm = np.full((nLabels, nLabels), 0)
    for i in range(nLabels):
        for j in range(nLabels):
            cm[i,j] = np.sum(np.where(y_true == i, 1, 0) * np.where(y_pred==j, 1, 0))
    
    return cm

def confusion_matrix_accuracy(cm):
    return np.trace(cm) / np.sum(cm)