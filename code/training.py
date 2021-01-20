import math
import time
from torch import nn
import numpy as np
from batch import process_batch
import pprint

# Train one epoch
def train_epoch(model, train_data_loader, training_set_size, log_vars, gradient_accumulation_steps, optimizer, scheduler, device, get_fixed_predictions=False):
    # Set the model to train
    model = model.train()  
    training_set_losses = []
    training_set_instance_ids = []
    training_set_probs = {}
    training_set_pred_labels = {}
    training_set_real_labels = {}
    training_set_fixed_pred_labels = {}        
    batch_number = 0;
    num_batches = math.ceil(training_set_size / train_data_loader.batch_size)
    # To show estimated time of arrival (ETA) based on latest 10 batches.
    accum_time = 0
    show_eta = 0   
    print(f'log_vars: {log_vars}')
    # Initialize results 
    for task in log_vars:
        training_set_real_labels[task] = []
        training_set_pred_labels[task] = []
        training_set_probs[task] = []
        training_set_fixed_pred_labels[task] = []        
    # Process each batch        
    for batch in train_data_loader:
        batch_number += 1
        init_time = time.perf_counter()
        results_batch = process_batch(model, batch, log_vars, device) 
        training_set_instance_ids.extend(results_batch['instance_ids'])
        for task in log_vars:
            training_set_real_labels[task].extend(results_batch['real_labels'][task])
            training_set_pred_labels[task].extend(results_batch['pred_labels'][task]) 
            training_set_probs[task].extend(results_batch['probs'][task].tolist())         
        if gradient_accumulation_steps > 1:
            results_batch['loss'] = results_batch['loss'] / gradient_accumulation_steps
        training_set_losses.append(results_batch['loss'].item())            
        results_batch['loss'].backward()
        # Update parameters      
        batches_left = num_batches - batch_number        
        if batch_number % gradient_accumulation_steps == 0 or batches_left == 0:  
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        end_time = time.perf_counter()
        time_batch = end_time-init_time
        show_eta += 1
        # Show ETA every 50 batches.
        if show_eta == 50: 
            print(f'Processed batch {batch_number}/{num_batches} in {time_batch:0.2f} seconds')        
            eta = math.ceil((accum_time/show_eta) * batches_left / 60)
            show_eta = 0
            accum_time = 0
            print(f'Epoch ETA: {eta:d} minutes')
        else:
            accum_time += time_batch
    # Loss for training set in current epoch
    mean_training_loss = np.mean(training_set_losses)
    # How to fix the predictions depend on the dataset and the task.
    if get_fixed_predictions:
        for task in log_vars:    
            training_set_fixed_pred_labels[task] = train_data_loader.dataset.get_fixed_predictions(training_set_instance_ids, training_set_probs, task)
    return {
        'loss': mean_training_loss,
        'instance_ids': training_set_instance_ids,
        'probs': training_set_probs,
        'real_labels': training_set_real_labels,
        'pred_labels': training_set_pred_labels,
        'fixed_pred_labels': training_set_fixed_pred_labels        
    }
    
    