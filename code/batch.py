import torch
from torch import nn
from config import config
import pprint

def criterion(batch_task_losses, log_vars):
    loss = 0
    for task in log_vars:
        precision = torch.exp(-log_vars[task])
        loss += torch.sum(precision*batch_task_losses[task] + log_vars[task], -1)
    return torch.mean(loss)


def process_batch(model, batch, log_vars, device):
    batch_instance_ids = batch['instance_id']
    batch_labels = batch['labels']
    batch_real_labels = {}        
    if len(batch_labels) > 0:
        for task in batch_labels:
            batch_labels[task] = batch_labels[task].to(device)
            batch_real_labels[task] = batch_labels[task].tolist()
    else:
        for task in batch_labels:    
            batch_real_labels[task] = []
    # Get predictions by model for each task.
    batch_input_ids = batch['input_ids'].to(device)
    batch_attention_masks = batch['attention_mask'].to(device)             
    batch_pred_labels = {}
    batch_probs = {}
    batch_task_logits = {}
    batch_pred_logits = {}    
    batch_task_losses = {}    
    for task in log_vars:
        batch_output = model(task, input_ids=batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels[task], return_dict=False)        
        # Output is a tuple (batch_logits, grad_fn)
        batch_task_logits[task] = batch_output[0]    
        batch_task_losses[task] = batch_output[1] 
        batch_pred_logits[task], batch_pred_labels[task] = torch.max(batch_task_logits[task], dim=1)  
        batch_probs[task] = nn.functional.softmax(batch_task_logits[task], dim=1)        
        batch_pred_logits[task] = batch_pred_logits[task].to(device)
        batch_pred_labels[task] = batch_pred_labels[task].tolist()
    # Compute global loss (multi-task)
    if batch_task_losses:
        if len(batch_labels) > 1:
            batch_loss = criterion(batch_task_losses, log_vars)
        else:
            batch_loss = list(batch_task_losses.values())[0]
    else:
        batch_loss =  torch.tensor(0.0, requires_grad=True)    
    return {
        'instance_ids': batch_instance_ids,
        'loss': batch_loss,        
        'logits': batch_task_logits,
        'real_labels': batch_real_labels,        
        'pred_labels': batch_pred_labels,
        'task_losses': batch_task_losses,
        'probs': batch_probs
    }

