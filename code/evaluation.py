import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef
from batch import process_batch
      
# Evaluate model / make predictions
# If no labels are available, real_labels=[] and loss=0 and can be ignored.
def predict(model, eval_data_loader, log_vars, device, get_fixed_predictions=False):
    # Set the model to eval
    model = model.eval()
    val_set_instance_ids = []
    val_set_losses = []
    val_set_pred_labels = {}
    val_set_real_labels = {}
    val_set_probs = {}   
    # Initialize results 
    for task in log_vars:
        val_set_real_labels[task] = []
        val_set_pred_labels[task] = []
        val_set_probs[task] = []
    with torch.no_grad():
        for batch in eval_data_loader:
            results_batch = process_batch(model, batch, log_vars, device)        
            val_set_instance_ids.extend(results_batch['instance_ids'])
            val_set_losses.append(results_batch['loss'].item())
            for task in log_vars:            
                val_set_real_labels[task].extend(results_batch['real_labels'][task])
                val_set_pred_labels[task].extend(results_batch['pred_labels'][task]) 
                val_set_probs[task].extend(results_batch['probs'][task].tolist())         
    # Loss for evaluation set            
    mean_val_loss = np.mean(val_set_losses)        
    # How to fix the predictions depend on the dataset.
    val_set_fixed_pred_labels = []
    if get_fixed_predictions:
        val_set_fixed_pred_labels = val_data_loader.dataset.get_fixed_predictions(val_set_instance_ids, val_set_probs)
    return {
        'loss': mean_val_loss,
        'instance_ids': val_set_instance_ids,
        'probs': val_set_probs,
        'real_labels': val_set_real_labels,
        'pred_labels': val_set_pred_labels,
        'fixed_pred_labels': val_set_fixed_pred_labels        
    }    

# Get metrics
def get_metrics(pred_labels, real_labels):
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(real_labels, pred_labels, average='weighted')    
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(real_labels, pred_labels, average='macro')    
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(real_labels, pred_labels, average='micro')        
    accuracy = accuracy_score(real_labels, pred_labels)
    mcc = matthews_corrcoef(real_labels, pred_labels)    
    return {
        'acc': accuracy,
        'w_f1': weighted_f1,
        'w_p': weighted_precision,
        'w_r': weighted_recall,
        'mic_f1': micro_f1,
        'mic_p': micro_precision,
        'mic_r': micro_recall,
        'mac_f1': macro_f1,
        'mac_p': macro_precision,
        'mac_r': macro_recall,
        'mcc': mcc
    }  
