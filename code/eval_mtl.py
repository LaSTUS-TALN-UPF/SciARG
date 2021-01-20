#!/usr/bin/python3
from transformers import BertTokenizer
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import sys
import os
import pprint
from distutils.util import strtobool
from arguments import args
from config import config
from evaluation import predict, get_metrics
from training import train_epoch
from output import show_results, show_results_by_annotator, print_confusion_matrix
#from batch import loss_function
from datasets import SciAbstractsDataset
from model import BertWithFeatForSequenceClass

# Display arguments
argparse_dict = vars(args)
print('--------------------------- Arguments ---------------------------')
pprint.pprint(argparse_dict)
print('-----------------------------------------------------------------\n')

if args.task:
    tasks = [args.task]
else:
    tasks = config.keys()
str_tasks = '_'.join(tasks)

if args.mode == 'background':
    log_dir = args.base_dir_logs + '/' + str_tasks
    log_file = log_dir + '/output_eval_' + args.date_time + '.log'    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)    
    print(f'\n\n ==> Redirecting output to:   {log_file}\n')
    sys.stdout = open(log_file, 'w')    

# Random seed for reproducibility
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

# Device
if torch.cuda.is_available():
    print('\n=> Device: cuda:0\n')
    device = torch.device("cuda:0")
else:
    print('\n=> Device: cpu !!!\n')
    device = torch.device("cpu")

# Text fields
text_fields_pairs = [f.strip() for f in args.text_fields.split('|')]
text_fields_single = [text_fields_pairs[0]]

# Labels fields
label_fields = {}
for task in tasks:
    print(f'TASK: {task}')
    label_fields[task] = config[task]['label_field']

# Additional features
# This could be task-dependant.
additional_features_fields = []
additional_features_tokens = {}
if args.additional_features:
    list_features = [f.strip() for f in args.additional_features.split(',')] if args.additional_features else []
    for feature in list_features:
        [column_name, token_name] = [f.strip() for f in feature.split(':')]
        additional_features_tokens[column_name] = token_name
        additional_features_fields.append(column_name)
    additional_features_fields = list(set(additional_features_fields)) 
    additional_features_fields.sort()

# Position fields
position_fields_pairs = [f.strip() for f in args.position_fields.split('|') if f] if args.position_fields else []
position_fields_single = [position_fields_pairs[0]]

# ----------------------------------------------------------------------------------------------------------
# -------------------------------------------------- Tokenizer ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=bool(strtobool(args.do_lower_case)))

# ----------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- Model -----------------------------------------------
# ----------------------------------------------------------------------------------------------------------
print(f'Loading model {args.model_name_or_path}')
model = BertWithFeatForSequenceClass.load_model(args.model_name_or_path)

model.to_device(device)
# Is this necessary?
model.resize_token_embeddings(len(tokenizer))

label_dict = model.label_dict
label_names = {}
label_ids = {}
for task_name in label_dict:
    label_names[task_name] = list(label_dict[task_name].keys())
    label_ids[task_name] = [i for i in range(len(label_names[task_name]))]


      
# ----------------------------------------------------------------------------------------------------------
# -------------------------------------------------- Data load ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# Instance fields
# Not filtering columns now.

columns_output = [f.strip() for f in args.columns_output.split(',') if f] if args.columns_output else []            
df_test_set = pd.read_csv(args.test_set_path, '\t', quoting=csv.QUOTE_NONE, encoding='utf-8')

# Filter test data by annotator(s)
if args.test_annotators:
    test_annotators = [f.strip() for f in args.test_annotators.split(',')]
elif args.annotators:
    test_annotators = [f.strip() for f in args.annotators.split(',')]   
else:
    test_annotators = []
if test_annotators:    
    print(f'Test annotators: {test_annotators}')    
    cond_annotators = df_test_set['annotator'].isin(test_annotators)
    df_test_set = df_test_set[cond_annotators]
    df_test_set.reset_index(inplace=True, drop=True)

# Make sure we don't have duplicated instances.
df_test_set__eval_single = df_test_set.drop_duplicates(subset=args.instance_id_field_single)
df_test_set__eval_pairs = df_test_set.drop_duplicates(subset=args.instance_id_field_pair)
df_test_set__eval_single.reset_index(inplace=True, drop=True)
df_test_set__eval_pairs.reset_index(inplace=True, drop=True)

# ----------------- DEBUG -------------
instances_test_eval_single = df_test_set__eval_single[args.instance_id_field_single]
instances_test_eval_pairs = df_test_set__eval_pairs[args.instance_id_field_pair]
print('\n====================  Instances train/test ==============================')
print(f'instances test  - eval single - {len(instances_test_eval_single)}')
print(instances_test_eval_single)
print()
print(f'instances test  - eval pairs - {len(instances_test_eval_pairs)}')
print(instances_test_eval_pairs)
print('==================================================\n')
#-------------------------------------

# Annotators (to show)
test_annotators_eval = {}
for task in tasks:
    if config[task]['single_sent']:
        test_annotators_eval[task] = list(df_test_set__eval_single['annotator'].unique()) if args.show_annotator_field else []
    else:
        test_annotators_eval[task] = list(df_test_set__eval_pairs['annotator'].unique()) if args.show_annotator_field else [] 

  
# ----------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- Datasets --------------------------------------------
# ----------------------------------------------------------------------------------------------------------
dataset_test_set__eval_pairs = SciAbstractsDataset(df_test_set__eval_pairs, tasks, text_fields_pairs, label_fields, position_fields_pairs, bool(strtobool(args.add_distance_tokens)), bool(strtobool(args.add_order_tokens)), additional_features_tokens, args.instance_id_field_single, args.instance_id_field_pair, label_dict, args.maxlen, tokenizer)
if args.encoding_single == 'different':
    dataset_test_set__eval_single = SciAbstractsDataset(df_test_set__eval_single, tasks, text_fields_single, label_fields, position_fields_single, bool(strtobool(args.add_distance_tokens)), bool(strtobool(args.add_order_tokens)), additional_features_tokens, args.instance_id_field_single, args.instance_id_field_pair, label_dict, args.maxlen, tokenizer)
else:
    dataset_test_set__eval_single = SciAbstractsDataset(df_test_set__eval_single, tasks, text_fields_pairs, label_fields, position_fields_pairs, bool(strtobool(args.add_distance_tokens)), bool(strtobool(args.add_order_tokens)), additional_features_tokens, args.instance_id_field_single, args.instance_id_field_pair, label_dict, args.maxlen, tokenizer)

# Check whether fixed predictions method is implemented - using the test set database.
'''
if bool(strtobool(args.show_fixed_predictions)):
    fixed_predictions_available = callable(getattr(dataset_test_set__eval_pairs, 'get_fixed_predictions', None))
    if not fixed_predictions_available:
        print(f'\n===> Predictions cannot be fixed as the "get_fixed_predictions" method is not implemented in {dataset_test_set__eval_pairs.__class__.__name__} <===\n')
        args.show_fixed_predictions = 'False'
'''        

# ----------------------------------------------------------------------------------------------------------       
# ---------------------------------------------------- Data loaders ----------------------------------------
# ----------------------------------------------------------------------------------------------------------
data_loader_test_set__eval_single = DataLoader(dataset=dataset_test_set__eval_single, batch_size=args.eval_batch_size, num_workers=args.num_workers)
data_loader_test_set__eval_pairs = DataLoader(dataset=dataset_test_set__eval_pairs, batch_size=args.eval_batch_size, num_workers=args.num_workers)

# ----------------------------------------------------------------------------------------------------------        
# ----------------------------------------------------- Evaluate -------------------------------------------
# ----------------------------------------------------------------------------------------------------------  
log_vars = {}
for task in tasks:
    log_vars[task] = torch.zeros((1,)).to(device)

print()
print(f'label_names: {label_names}')
print(f'label_dict: {label_dict}')
print(f'label_ids: {label_ids}')
 

print()
    
# Evaluate test set.
for task in tasks:   
    if config[task]['single_sent']:
        test_set_results__eval = predict(model, data_loader_test_set__eval_single, log_vars, device, get_fixed_predictions=bool(strtobool(args.show_fixed_predictions)))
    else:
        test_set_results__eval = predict(model, data_loader_test_set__eval_pairs, log_vars, device, get_fixed_predictions=bool(strtobool(args.show_fixed_predictions)))
  
    print(f'\n========================= TASK: {task} ================================')    
    test_set_metrics = get_metrics(test_set_results__eval['pred_labels'][task], test_set_results__eval['real_labels'][task]) 
    show_number_instances = len(test_set_results__eval['pred_labels'][task])                        
    show_results('Test ('+str(show_number_instances)+' instances)', test_set_results__eval, task, test_set_metrics, annotators=','.join(test_annotators_eval[task]), show_fixed_predictions=bool(strtobool(args.show_fixed_predictions)))
            
    print('====================================================================') 
    print('  Report - Original predictions')
    print(classification_report(y_true=test_set_results__eval['real_labels'][task], y_pred=test_set_results__eval['pred_labels'][task], labels=label_ids[task], target_names=label_names[task]))
    print('====================================================================')            

    if bool(strtobool(args.show_fixed_predictions)) and test_set_results__eval['fixed_eval_labels'][task]:
        print('====================================================================') 
        print('  Report - Fixed predictions')
        print(classification_report(y_true=test_set_results__eval['real_labels'][task], y_pred=test_set_results__eval['fixed_eval_labels'][task], labels=label_ids, target_names=label_names))
        print('====================================================================')            

    # Show confusion matrix
    if bool(strtobool(args.show_confusion_matrix)):
        test_set_real_label_names = [label_names[task][i] for i in test_set_results__eval['real_labels'][task]]
        test_set_eval_label_names = [label_names[task][i] for i in test_set_results__eval['pred_labels'][task]]
        print_confusion_matrix(test_set_real_label_names, test_set_eval_label_names, label_names[task], hide_diagonal=False)

    '''
    # Show detailed predictions for each instance
    if bool(strtobool(args.show_detail_predictions)):
        show_detail_predictions(args.task, label_names, test_set_results__eval['pred_labels'][task], test_set_results__eval['real_labels'][task], test_set_results__eval['probs'][task], test_set_results__eval['instance_ids'], 'Predictions')         
        if bool(strtobool(args.show_fixed_predictions)) and test_set_results__eval['fixed_eval_labels'][task]:
            show_detail_predictions(args.task, label_names, test_set_results__eval['fixed_eval_labels'][task], test_set_results__eval['real_labels'][task], test_set_results__eval['probs'][task], test_set_results__eval['instance_ids'], 'Fixed Predictions')         
    '''




