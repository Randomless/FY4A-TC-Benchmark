import itertools
import torch
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

def compute_accuracy(y_pred, y_true,pred_type='regression'):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.data.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.data.cpu().numpy()
    if pred_type=='classification':
        if y_pred.shape[1] == 1:
            y_pred = y_pred[:, 0]
        else:
            y_pred = np.argmax(y_pred, axis=1)
        return metrics.accuracy_score(y_true, y_pred, normalize=True)
    elif pred_type=='regression':
        return  metrics.mean_squared_error(y_true, y_pred)#, metrics.mean_absolute_error(y_true, y_pred)#, metrics.r2_score(y_true, y_pred)
    elif pred_type=='multitask':
        return metrics.mean_squared_error(y_true, y_pred)#, metrics.mean_absolute_error(y_true, y_pred), metrics.r2_score(y_true, y_pred)
    else:
        raise ValueError('pred_type must be either classification, regression or multitask')

def compute_accuracy_patch(output, target=None,speed=None,pred_type='regression'):
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    if torch.is_tensor(speed):
        speed = speed.data.cpu().numpy()

    if pred_type=='classification':
        if output.shape[1] == 1:
            output = output[:, 0]
        else:
            output = np.argmax(output, axis=1)
        return metrics.accuracy_score(target, output, normalize=True)
    elif pred_type=='regression':
        return  metrics.mean_squared_error(speed, output),metrics.mean_absolute_error(speed,output)#, metrics.mean_absolute_error(target, output)#, metrics.r2_score(target, output)
    elif pred_type=='multitask':
        return metrics.mean_squared_error(speed, output[:, 0]),metrics.mean_absolute_error(speed,output[:, 0])#, metrics.mean_absolute_error(target, output), metrics.r2_score(target, output)
    else:
        raise ValueError('pred_type must be either classification, regression or multitask')

def compute_f1(y_pred, y_true):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.data.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.data.cpu().numpy()
    if y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0]
    else:
        y_pred = np.argmax(y_pred, axis=1)
    # 'micro', 'macro', 'weighted'
    return metrics.f1_score(y_true, y_pred, average='weighted')


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure
