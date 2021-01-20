#!/usr/bin/python3
from transformers import BertTokenizer
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
import numpy as np
import sys
import os
import pprint
from distutils.util import strtobool
from arguments import args
from config import config
from evaluation import predict
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
    log_file = log_dir + '/output_pred_' + args.date_time + '.log'    
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
for task_name in label_dict:
    label_names[task_name] = list(label_dict[task_name].keys())
    
# ----------------------------------------------------------------------------------------------------------
# -------------------------------------------------- Data load ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# Instance fields
# Not filtering columns now.

columns_output = [f.strip() for f in args.columns_output.split(',') if f] if args.columns_output else []            
df_test_set = pd.read_csv(args.test_set_path, '\t', quoting=csv.QUOTE_NONE, encoding='utf-8')

# Remove label fields. Copied to another column if they are to be included in the output.
for task in tasks:
    if label_fields[task] in columns_output:
        # We save the values to show them in the predictions.
        real_label_field = 'real_' + label_fields[task]
        df_test_set[real_label_field] = df_test_set[label_fields[task]]        
    df_test_set[label_fields[task]] = label_names[task][0]

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
df_test_set__pred_single = df_test_set.drop_duplicates(subset=args.instance_id_field_single)
df_test_set__pred_pairs = df_test_set.drop_duplicates(subset=args.instance_id_field_pair)
df_test_set__pred_single.reset_index(inplace=True, drop=True)
df_test_set__pred_pairs.reset_index(inplace=True, drop=True)

# ----------------- DEBUG -------------
instances_test_eval_single = df_test_set__pred_single[args.instance_id_field_single]
instances_test_eval_pairs = df_test_set__pred_pairs[args.instance_id_field_pair]
print('\n====================  Instances train/test ==============================')
print(f'instances test  - eval single - {len(instances_test_eval_single)}')
print(instances_test_eval_single)
print()
print(f'instances test  - eval pairs - {len(instances_test_eval_pairs)}')
print(instances_test_eval_pairs)
print('==================================================\n')
#-------------------------------------
  
# ----------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- Datasets --------------------------------------------
# ----------------------------------------------------------------------------------------------------------
dataset_test_set__pred_pairs = SciAbstractsDataset(df_test_set__pred_pairs, tasks, text_fields_pairs, label_fields, position_fields_pairs, bool(strtobool(args.add_distance_tokens)), bool(strtobool(args.add_order_tokens)), additional_features_tokens, args.instance_id_field_single, args.instance_id_field_pair, label_dict, args.maxlen, tokenizer)
if args.encoding_single == 'different':
    dataset_test_set__pred_single = SciAbstractsDataset(df_test_set__pred_single, tasks, text_fields_single, label_fields, position_fields_single, bool(strtobool(args.add_distance_tokens)), bool(strtobool(args.add_order_tokens)), additional_features_tokens, args.instance_id_field_single, args.instance_id_field_pair, label_dict, args.maxlen, tokenizer)
else:
    dataset_test_set__pred_single = SciAbstractsDataset(df_test_set__pred_single, tasks, text_fields_pairs, label_fields, position_fields_pairs, bool(strtobool(args.add_distance_tokens)), bool(strtobool(args.add_order_tokens)), additional_features_tokens, args.instance_id_field_single, args.instance_id_field_pair, label_dict, args.maxlen, tokenizer)

# Check whether fixed predictions method is implemented - using the test set database.
'''
if bool(strtobool(args.show_fixed_predictions)):
    fixed_predictions_available = callable(getattr(dataset_test_set__pred_pairs, 'get_fixed_predictions', None))
    if not fixed_predictions_available:
        print(f'\n===> Predictions cannot be fixed as the "get_fixed_predictions" method is not implemented in {dataset_test_set__pred_pairs.__class__.__name__} <===\n')
        args.show_fixed_predictions = 'False'
'''        

# ----------------------------------------------------------------------------------------------------------       
# ---------------------------------------------------- Data loaders ----------------------------------------
# ----------------------------------------------------------------------------------------------------------
data_loader_test_set__pred_single = DataLoader(dataset=dataset_test_set__pred_single, batch_size=args.eval_batch_size, num_workers=args.num_workers)
data_loader_test_set__pred_pairs = DataLoader(dataset=dataset_test_set__pred_pairs, batch_size=args.eval_batch_size, num_workers=args.num_workers)

# ----------------------------------------------------------------------------------------------------------        
# ------------------------------------------------------ Predict -------------------------------------------
# ----------------------------------------------------------------------------------------------------------
log_vars = {}
for task in tasks:
    log_vars[task] = torch.zeros((1,)).to(device)
    
# Predict and save dataframe with predictions.        
for task in tasks:   
    if config[task]['single_sent']:
        test_set_results__pred = predict(model, data_loader_test_set__pred_single, log_vars, device, get_fixed_predictions=bool(strtobool(args.show_fixed_predictions)))
        df_output = df_test_set__pred_single.copy()
    else:
        test_set_results__pred = predict(model, data_loader_test_set__pred_pairs, log_vars, device, get_fixed_predictions=bool(strtobool(args.show_fixed_predictions)))
        df_output = df_test_set__pred_pairs.copy()
    if label_fields[task] in columns_output:
        columns_output.append(real_label_field)        
    pred_label_field = 'pred_' + label_fields[task]
    predicted_labels = test_set_results__pred['pred_labels'][task]
    predicted_probs = test_set_results__pred['probs'][task]
    df_output[pred_label_field] = [label_names[task][pred_label] for pred_label in predicted_labels]
    columns_output.append(pred_label_field)            
    if bool(strtobool(args.show_probabilities)):
        for label in label_dict[task]:
            df_output[label] = [predictions[label_dict[task][label]] for predictions in predicted_probs]
            columns_output.append(label)
    if args.save_predictions_path:
        dir_path = os.path.dirname(os.path.realpath(args.save_predictions_path))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)            
        df_output.to_csv(args.save_predictions_path, '\t', quoting=csv.QUOTE_NONE, encoding='utf-8', columns=columns_output, index=False)
        print()
        print(f'===> Predictions saved at {args.save_predictions_path}')
    
    



