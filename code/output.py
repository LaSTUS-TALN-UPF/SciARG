from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from evaluation import get_metrics

def show_results(set_name, results, task, metrics, annotators='', show_fixed_predictions=False, epoch=0, num_instances=0):
    show_epoch = f'- Epoch: {epoch}' if epoch else ''
    show_annotators = f'- Annotators: {annotators}' if annotators else ''
    print('\n====================================================================')    
    print(f'  {set_name} set {show_epoch} {show_annotators} - {num_instances} instances')
    print('====================================================================')
    print(f'  {set_name} Loss: {results["loss"]:0.4f} - Task: {task}')
    print(f'  {set_name} Accuracy (Micro F1): {metrics["acc"]:0.4f} {show_epoch} {show_annotators} - Task: {task}')
    print(f'  {set_name} MCC: {metrics["mcc"]:0.4f} {show_epoch} {show_annotators} - Task: {task}')
    print(f'  {set_name} Weighted P, R, F1: {metrics["w_p"]:0.4f} {metrics["w_r"]:0.4f} {metrics["w_f1"]:0.4f} {show_epoch} {show_annotators} - Task: {task}')
    print(f'  {set_name} Macro P, R, F1: {metrics["mac_p"]:0.4f} {metrics["mac_r"]:0.4f} {metrics["mac_f1"]:0.4f} {show_epoch} {show_annotators} - Task: {task}')
    # Fixed predictions
    if show_fixed_predictions and results['fixed_pred_labels'][task]:
        fixed_metrics = get_metrics(results['fixed_pred_labels'][task], results['real_labels'][task])
        print('--------------------------------------------------------------------')        
        print(f'    {set_name} set - Fixed predictions {show_epoch} {show_annotators}')
        print('--------------------------------------------------------------------')
        print(f'    Fixed {set_name} Accuracy (Micro F1): {fixed_metrics["acc"]:0.4f} {show_epoch} {show_annotators} - Task: {task}')
        print(f'    Fixed {set_name} MCC: {metrics["mcc"]:0.4f} {show_epoch} {show_annotators} - Task: {task}')
        print(f'    Fixed {set_name} Weighted P, R, F1: {fixed_metrics["w_p"]:0.4f} {fixed_metrics["w_r"]:0.4f} {fixed_metrics["w_f1"]:0.4f} {show_epoch} {show_annotators} - Task: {task}')
        print(f'    Fixed {set_name} Macro P, R, F1: {fixed_metrics["mac_p"]:0.4f} {fixed_metrics["mac_r"]:0.4f} {fixed_metrics["mac_f1"]:0.4f} {show_epoch} {show_annotators} - Task: {task}')
    #print('====================================================================')

def show_results_by_annotator(set_name, results_annotators, metrics_annotators, epoch=0):
    show_epoch = f'- Epoch: {epoch}' if epoch else ''
    print('────────────────────────────────────────────────────────────────────')
    print(f'        {set_name} set metrics by annotator {show_epoch}')
    for ann in results_annotators:        
        print('────────────────────────────────────────────────────────────────────')   
        print(f'        {set_name} Loss: {results_annotators[ann]["loss"]:0.4f} {show_epoch} - Annotator: {ann} - Task: {task}')
        print(f'        {set_name} Accuracy (Micro F1): {metrics_annotators[ann]["acc"]:0.4f} {show_epoch} - Annotator: {ann} - Task: {task}')
        print(f'        {set_name} MCC: {metrics["mcc"]:0.4f} {show_epoch} {show_annotators} - Task: {task}')
        print(f'        {set_name} Weighted P, R, F1: {metrics_annotators[ann]["w_p"]:0.4f} {metrics_annotators[ann]["w_r"]:0.4f} {metrics_annotators[ann]["w_f1"]:0.4f} {show_epoch} - Annotator: {ann} - Task: {task}')
        print(f'        {set_name} Macro P, R, F1: {metrics_annotators[ann]["mac_p"]:0.4f} {metrics_annotators[ann]["mac_r"]:0.4f} {metrics_annotators[ann]["mac_f1"]:0.4f} {show_epoch} - Annotator: {ann} - Task: {task}')
    print('────────────────────────────────────────────────────────────────────')   

# Print confusion matrix
def print_confusion_matrix(real_labels, pred_labels, labels=None, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    if labels is None:
        labels = unique_labels(real_labels, pred_labels)    
    cm = confusion_matrix(real_labels, pred_labels, labels=labels)
    # find which fixed column width will be used for the matrix
    columnwidth = max([len(str(x)) for x in labels])  # longest class name
    empty_cell = ' ' * columnwidth
    print('T/P' + ' ' * (columnwidth - 3), end='\t')   
    for label in labels:
        print(f'{label:^{columnwidth}}', end='\t')  # right-aligned label padded with spaces to columnwidth
    print()  # newline
    # Print rows
    for i, label in enumerate(labels):
        print(f'{label:{columnwidth}}', end='\t')  # label padded with spaces to columnwidth
        for j in range(len(labels)):
            # cell value padded to columnwidth with spaces and displayed with 1 decimal
            cell = f'{cm[i, j]:^{columnwidth}}'
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end='\t')
        print()    
        
'''
# Show predictions for every instance        
def show_detail_predictions(task, label_names, pred_labels, real_labels, test_set_probs, instances_ids, title='Predictions'):
    print('\n====================================================================')    
    print(f'                 {title}')
    print('====================================================================\n')        
    # Show header
    if task == 'parents_pair':
        print('\n{:3}\t{:10}\t{:4}\t{:4}\t{:3}\t{:4}\t{:4}\t{:6}\t{}'.format('DOM', 'ID', 'ADU1', 'ADU2', 'ANN', 'REAL', 'PRED', 'PROB', 'CORRECT'))
    else:
        print('\n{:3}\t{:10}\t{:3}\t{:3}\t{:24}\t{:24}\t{:6}\t{}'.format('DOM', 'ID', 'ADU', 'ANN', 'REAL', 'PRED', 'PROB', 'CORRECT'))
    # Show predictions
    for i in range(len(instances_ids)):
        pred_label = label_names[pred_labels[i]]
        real_label = label_names[real_labels[i]]    
        correct = 'Y' if real_label == pred_label else 'N'
        instance_id =  instances_ids[i].split('__')
        if task == 'parents_pair':
            [dom, id, adu1, adu2, ann] =  instance_id
            print(f'{dom:3}\t{id:10}\t{adu1:4}\t{adu2:4}\t{ann:3}\t{real_label:4}\t{pred_label:4}\t{max(test_set_probs[i]):0.4f}\t{correct}')   
        else:
            [dom, id, adu, ann] =  instance_id
            print(f'{dom:3}\t{id:10}\t{adu:3}\t{ann:4}\t{real_label:24}\t{pred_label:24}\t{max(test_set_probs[i]):0.4f}\t{correct}')        
    print('====================================================================\n')        
'''

