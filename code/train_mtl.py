#!/usr/bin/python3
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import math
import pprint
import statistics
import re
import sys
import os
from distutils.util import strtobool
from arguments import args
from config import config
from evaluation import predict, get_metrics
from training import train_epoch
from output import show_results, show_results_by_annotator
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
    tasks = list(config.keys())
    
print(f'TASKS: {tasks}')
str_tasks = '_'.join(tasks)

if args.gradient_accumulation_steps < 1:
    args.gradient_accumulation_steps = 1

if args.mode == 'background':
    log_dir = args.base_dir_logs + '/' + str_tasks
    train_shuffle = 'shuffle_' if bool(strtobool(args.shuffle)) else ''
    log_file = log_dir + '/output_train_' + train_shuffle + '_' + args.date_time + '.log'    
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

# Columns in dataframes
df_columns = []
df_columns.extend(text_fields_pairs)

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
    df_columns.extend(additional_features_fields)  

# ----------------------------------------------------------------------------------------------------------
# -------------------------------------------------- Data load ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# Instance fields
df_columns.append(args.instance_id_field_single)
df_columns.append(args.instance_id_field_pair)

# Annotator field
if args.show_annotator_field:
    df_columns.append(args.show_annotator_field)
    
# Position fields
position_fields_pairs = [f.strip() for f in args.position_fields.split('|') if f] if args.position_fields else []
position_fields_single = [position_fields_pairs[0]]
df_columns.extend(position_fields_pairs)

# Labels fields
label_fields = {}
label_names = {}    
for task in tasks:
    label_fields[task] = config[task]['label_field']

# Read data.
# Training is done with pair of sentences.

# Add label fields and delete duplicates in column names.
df_columns_train = list(set(df_columns + list(label_fields.values())))
print(f'df_columns_train: {df_columns_train}')
df_training_set__train_all = pd.read_csv(args.training_set_path, '\t', quoting=csv.QUOTE_NONE, encoding='utf-8', usecols=df_columns_train)

# Labels in training set (before CV splitting)
for task in tasks:   
    label_field = label_fields[task]     
    label_names[task] = df_training_set__train_all[label_field].unique() 

# Additional values found in training set added as tokens.
additional_features_values = {field:df_training_set__train_all[field].unique() for field in additional_features_fields}
print('---------------------- Additional features ----------------------')
print('additional_features_values:')
pprint.pprint(additional_features_values)
print('-----------------------------------------------------------------\n') 

label_ids = {}
label_dict = {}
for task in tasks:
    label_names[task] = sorted(label_names[task])    
    label_ids[task] = [i for i in range(len(label_names[task]))]
    label_dict[task] = dict([(label_names[task][i], i) for i in label_ids[task]])

print('------------------------------- Labels --------------------------')
pprint.pprint(label_dict)
print('-----------------------------------------------------------------\n')  

# Filter training data by annotator(s)
if args.train_annotators:
    train_annotators = [f.strip() for f in args.train_annotators.split(',')]
elif args.annotators:
    train_annotators = [f.strip() for f in args.annotators.split(',')]
else:
    train_annotators = []
if train_annotators:
    print(f'Train annotators: {train_annotators}')                
    cond_annotators = df_training_set__train_all['annotator'].isin(train_annotators)
    df_training_set__train_all = df_training_set__train_all[cond_annotators]
    df_training_set__train_all.reset_index(inplace=True, drop=True)       

# Cross-validation splits.
# Stratified splits are done based on one task passed as argument.
if args.num_cv_folds and args.cv_fold < args.num_cv_folds:
    if args.cv_split_task in label_fields:
        cv_split_task = args.cv_split_task
    else:
        cv_split_task = tasks[0]
        print(f'Task {task} - label {label_fields[cv_split_task]} considered for cross-validation stratified split.')            
    cv_split_field = label_fields[cv_split_task]
    skf = StratifiedKFold(n_splits=args.num_cv_folds, random_state=args.random_seed)    
    indices_train_test = [(train_indices_split, test_indices_split) for train_indices_split, test_indices_split in skf.split(df_training_set__train_all, df_training_set__train_all[cv_split_field])]
    indices_train = indices_train_test[args.cv_fold][0]
    indices_test = indices_train_test[args.cv_fold][1]
    df_test_fold = df_training_set__train_all.iloc[indices_test].copy()
    df_training_set__train_all = df_training_set__train_all.iloc[indices_train]
    # Make sure test set instances are not in the training set - for single and pairs respectively.
    df_test_set__eval_pairs = df_test_fold[~df_test_fold[args.instance_id_field_pair].isin(df_training_set__train_all[args.instance_id_field_pair])].copy()
    df_training_set__train_all.reset_index(inplace=True, drop=True)
else:
    # Not cross-validation - load test set.
    # When training we include label values for evaluation in test set. so we use df_columns_train.
    df_test_set__eval_pairs = pd.read_csv(args.test_set_path, '\t', quoting=csv.QUOTE_NONE, encoding='utf-8', usecols=df_columns_train)
    # Filter test data by annotator(s)
    if args.test_annotators:
        test_annotators = [f.strip() for f in args.test_annotators.split(',')]
    elif args.annotators:
        test_annotators = [f.strip() for f in args.annotators.split(',')]   
    else:
        test_annotators = []
    if test_annotators:    
        print(f'Test annotators: {test_annotators}')    
        cond_annotators = df_test_set__eval_pairs['annotator'].isin(test_annotators)
        df_test_set__eval_pairs = df_test_set__eval_pairs[cond_annotators]
        df_test_set__eval_pairs.reset_index(inplace=True, drop=True)

# Make sure we don't have duplicated instances.
df_test_set__eval_single = df_test_set__eval_pairs.drop_duplicates(subset=args.instance_id_field_single)
df_test_set__eval_pairs = df_test_set__eval_pairs.drop_duplicates(subset=args.instance_id_field_pair)
df_test_set__eval_single.reset_index(inplace=True, drop=True)
df_test_set__eval_pairs.reset_index(inplace=True, drop=True)

# For evaluation in training set we keep only one instance - based on adu id or relation id, depending on the task.
df_training_set__eval_single = df_training_set__train_all.drop_duplicates(subset=args.instance_id_field_single)
df_training_set__eval_pairs = df_training_set__train_all.drop_duplicates(subset=args.instance_id_field_pair)
df_training_set__eval_single.reset_index(inplace=True, drop=True)
df_training_set__eval_pairs.reset_index(inplace=True, drop=True)

# ----------------- DEBUG -------------
instances_train_train_all = df_training_set__train_all[args.instance_id_field_pair]
instances_train_eval_single = df_training_set__eval_single[args.instance_id_field_single]
instances_train_eval_pairs = df_training_set__eval_pairs[args.instance_id_field_pair]
instances_test_eval_single = df_test_set__eval_single[args.instance_id_field_single]
instances_test_eval_pairs = df_test_set__eval_pairs[args.instance_id_field_pair]
print('\n====================  Instances train/test ==============================')
print(f'instances train - train all - {len(instances_train_train_all)}')
print(instances_train_train_all)
print()
print(f'instances train - eval single - {len(instances_train_eval_single)}')
print(instances_train_eval_single)
print()
print(f'instances train - eval pairs - {len(instances_train_eval_pairs)}')
print(instances_train_eval_pairs)
print()
print(f'instances test  - eval single - {len(instances_test_eval_single)}')
print(instances_test_eval_single)
print()
print(f'instances test  - eval pairs - {len(instances_test_eval_pairs)}')
print(instances_test_eval_pairs)
print('==================================================\n')
#-------------------------------------

# Annotators (to show)
train_annotators_train = list(df_training_set__train_all['annotator'].unique()) if args.show_annotator_field else []
train_annotators_eval = {}
test_annotators_eval = {}
for task in tasks:
    if config[task]['single_sent']:
        train_annotators_eval[task] = list(df_training_set__eval_single['annotator'].unique()) if args.show_annotator_field else []
        test_annotators_eval[task] = list(df_test_set__eval_single['annotator'].unique()) if args.show_annotator_field else []
    else:
        train_annotators_eval[task] = list(df_training_set__eval_pairs['annotator'].unique()) if args.show_annotator_field else []
        test_annotators_eval[task] = list(df_test_set__eval_pairs['annotator'].unique()) if args.show_annotator_field else []   

# DEBUG
'''
print('--------------------------- Doc Ids ---------------------------')
training_set_ids = df_training_set__train_all['doc_id'].unique().tolist()
print(f'Training set: {training_set_ids}')
test_set_ids = df_test_set__eval_single['doc_id'].unique().tolist()
print(f'Test set: {test_set_ids}')
print('---------------------------------------------------------------\n')
'''
    
# Lengths / truncation
# TODO: Count additional tokens. Now adding extra 10 positions for special tokens, symbols, etc.
#       Check that it's not greater than the models' max.
# Get words and most frequent punctuation strings.

tokens_re = re.compile(r'(\w+|[\(\).?\-",%;]+)')
lengths_text_fields = []
for i in range(2):
    lengths_text_fields.append([len(tokens_re.findall(text)) for text in df_training_set__train_all[text_fields_pairs[i]] if len(text) > 0])
lengths_texts_train = [length_1 + length_2 for length_1, length_2 in zip(lengths_text_fields[0], lengths_text_fields[1])]

median_length_texts_train = statistics.median(lengths_texts_train)
avg_length_texts_train = math.ceil(statistics.mean(lengths_texts_train))
max_length_texts_train = max(lengths_texts_train)

print(f'Max length train: {max_length_texts_train}, Avg. length train: {avg_length_texts_train}, Median length train: {median_length_texts_train}')
# args.maxlen has to be INT - otherwise it raises a type error !!!
if args.maxlen == 0:
    args.maxlen = int(2 * median_length_texts_train) + 10;
    print(f'Max length set to {args.maxlen}\n')

# ----------------------------------------------------------------------------------------------------------
# -------------------------------------------------- Tokenizer ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=bool(strtobool(args.do_lower_case)), model_max_length=args.maxlen)

# Add special tokens for additional features.
additional_special_tokens = [t.strip() for t in args.special_tokens.split(',') if t]

# Additional tokens
if additional_features_tokens:
    for feature_field in additional_features_tokens:
        feature_token_prefix = additional_features_tokens[feature_field]
        for feature_value in additional_features_values[feature_field]:
            feature_token_value = str(feature_value).replace(' ', '_').upper()
            additional_token = '[' + feature_token_prefix + '_' + feature_token_value + ']'
            additional_special_tokens.append(additional_token)
    additional_special_tokens = list(set(additional_special_tokens))

if len(position_fields_pairs) > 1:
    # Distance tokens
    max_position = 0        
    if  bool(strtobool(args.add_distance_tokens)):
        for i in range(len(position_fields_pairs)):
            max_column = df_training_set__train_all[position_fields_pairs[i]].max()
            if max_column > max_position:
                max_position = max_column
    for i in range(1, max_position-1):
        additional_special_tokens.append('[DISTANCE_' + str(i) + ']')
    # Order tokens
    if bool(strtobool(args.add_order_tokens)):
        additional_special_tokens.extend(['[BEFORE]','[AFTER]'])
               
print('------------------------------ Additional tokens --------------------------')
print(additional_special_tokens) 
print('--------------------------------------------------------------------------\n')     
               
if additional_special_tokens:
    tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})

# ----------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- Model -----------------------------------------------
# ----------------------------------------------------------------------------------------------------------
pooled_tokens = [int(f.strip()) for f in args.pooled_tokens.split(',') if f]
print('------------------------------ Pooled tokens --------------------------')
print(pooled_tokens) 
print('-----------------------------------------------------------------------\n')  

model = BertWithFeatForSequenceClass.from_pretrained(args.model_name_or_path, hidden_dropout_prob=args.dropout, tasks=tasks, pooled_tokens=pooled_tokens, label_dict=label_dict)
#model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=len(label_dict), hidden_dropout_prob=args.dropout) 

if bool(strtobool(args.freeze_bert)):
    for param in model.bert.parameters():
        param.requires_grad = False

model.to_device(device)
model.resize_token_embeddings(len(tokenizer))

# ----------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- Datasets --------------------------------------------
# ----------------------------------------------------------------------------------------------------------
dataset_training_set__train_all = SciAbstractsDataset(df_training_set__train_all, tasks, text_fields_pairs, label_fields, position_fields_pairs, bool(strtobool(args.add_distance_tokens)), bool(strtobool(args.add_order_tokens)), additional_features_tokens, args.instance_id_field_single, args.instance_id_field_pair, label_dict, args.maxlen, tokenizer)
dataset_training_set__eval_pairs = SciAbstractsDataset(df_training_set__eval_pairs, tasks, text_fields_pairs, label_fields, position_fields_pairs, bool(strtobool(args.add_distance_tokens)), bool(strtobool(args.add_order_tokens)), additional_features_tokens, args.instance_id_field_single, args.instance_id_field_pair, label_dict, args.maxlen, tokenizer)
dataset_test_set__eval_pairs = SciAbstractsDataset(df_test_set__eval_pairs, tasks, text_fields_pairs, label_fields, position_fields_pairs, bool(strtobool(args.add_distance_tokens)), bool(strtobool(args.add_order_tokens)), additional_features_tokens, args.instance_id_field_single, args.instance_id_field_pair, label_dict, args.maxlen, tokenizer)
if args.encoding_single == 'different':
    dataset_training_set__eval_single = SciAbstractsDataset(df_training_set__eval_single, tasks, text_fields_single, label_fields, position_fields_single, bool(strtobool(args.add_distance_tokens)), bool(strtobool(args.add_order_tokens)), additional_features_tokens, args.instance_id_field_single, args.instance_id_field_pair, label_dict, args.maxlen, tokenizer)
    dataset_test_set__eval_single = SciAbstractsDataset(df_test_set__eval_single, tasks, text_fields_single, label_fields, position_fields_single, bool(strtobool(args.add_distance_tokens)), bool(strtobool(args.add_order_tokens)), additional_features_tokens, args.instance_id_field_single, args.instance_id_field_pair, label_dict, args.maxlen, tokenizer)
else:
    dataset_training_set__eval_single = SciAbstractsDataset(df_training_set__eval_single, tasks, text_fields_pairs, label_fields, position_fields_pairs, bool(strtobool(args.add_distance_tokens)), bool(strtobool(args.add_order_tokens)), additional_features_tokens, args.instance_id_field_single, args.instance_id_field_pair, label_dict, args.maxlen, tokenizer)
    dataset_test_set__eval_single = SciAbstractsDataset(df_test_set__eval_single, tasks, text_fields_pairs, label_fields, position_fields_pairs, bool(strtobool(args.add_distance_tokens)), bool(strtobool(args.add_order_tokens)), additional_features_tokens, args.instance_id_field_single, args.instance_id_field_pair, label_dict, args.maxlen, tokenizer)

# ----------------------------------------------------------------------------------------------------------       
# ---------------------------------------------------- Data loaders ----------------------------------------
# ----------------------------------------------------------------------------------------------------------
data_loader_training_set__train_all = DataLoader(dataset=dataset_training_set__train_all, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=bool(strtobool(args.shuffle)))
data_loader_training_set__eval_single = DataLoader(dataset=dataset_training_set__eval_single, batch_size=args.eval_batch_size, num_workers=args.num_workers)
data_loader_training_set__eval_pairs = DataLoader(dataset=dataset_training_set__eval_pairs, batch_size=args.eval_batch_size, num_workers=args.num_workers)
data_loader_test_set__eval_single = DataLoader(dataset=dataset_test_set__eval_single, batch_size=args.eval_batch_size, num_workers=args.num_workers)
data_loader_test_set__eval_pairs = DataLoader(dataset=dataset_test_set__eval_pairs, batch_size=args.eval_batch_size, num_workers=args.num_workers)

# ----------------------------------------------------------------------------------------------------------
# ---------------------------------------------------- Parameters ------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# Loss function in batch.py
#loss_function.to(device)

# Optimizer - Now using Transformers' versions.
accumulated_batch_size = args.gradient_accumulation_steps * args.train_batch_size
num_training_steps = math.ceil(args.epochs * len(df_training_set__train_all) / accumulated_batch_size)
num_warmup_steps = math.ceil(args.warmup_steps_percentage * num_training_steps)

log_vars = {}
for task in tasks:
    log_vars[task] = torch.zeros((1,)).to(device)

# get all parameters (model parameters + task dependent log variances)
# PASARLO AL MODELO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#params = [p for p in model.parameters()]
params = model.get_parameters()
for task in tasks:
    params.append(log_vars[task].requires_grad_())

#params.extend(list(log_vars.values()))
optimizer = AdamW(params, lr=args.lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False    
#optimizer = optim.Adam(params, lr=args.lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps) 

# Check whether fixed predictions method is implemented - using the test set database.
if bool(strtobool(args.show_fixed_predictions)):
    fixed_predictions_available = callable(getattr(dataset_test_set__eval_pairs, 'get_fixed_predictions', None))
    if not fixed_predictions_available:
        print(f'\n===> Predictions cannot be fixed as the "get_fixed_predictions" method is not implemented in {dataset_test_set__eval_pairs.__class__.__name__} <===\n')
        args.show_fixed_predictions = 'False'

# ----------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- Train --------------------------------------------
# ----------------------------------------------------------------------------------------------------------
output_path_last = {}
output_path_all_epochs = {}

# If training MTL we only save the model once with all the heads, for the first task.
first_task = tasks[0]
if len(tasks) > 1:
    task_name = str_tasks
else:
    task_name = first_task
    
output_path_last[first_task] = args.base_dir_models + '/' + task_name + '/' + args.model_name_or_path + '/' + 'epoch-' + str(args.epochs) + '/' + label_fields[task] + '__' + '-'.join(additional_features_fields) + '__' + str(args.lr) + '__' + str(args.train_batch_size) + '__'+ args.date_time
output_path_all_epochs[first_task] = args.base_dir_models + '/' + task_name + '/' + args.model_name_or_path + '/' + 'all_epochs' + '/' + label_fields[task] + '__' + '-'.join(additional_features_fields) + '__' + str(args.lr) + '__' + str(args.train_batch_size) + '__' + args.date_time
if not os.path.exists(output_path_last[first_task]):
    os.makedirs(output_path_last[first_task])

for epoch in range(args.epochs):
    epoch += 1
    '''
    # DEBUG - Show model's parameters in first epoch
    if epoch == 1:
        print('\n====================================================================')        
        print('Model parameters (training)')
        for parameter in model.parameters():
            print(parameter.size())
            print(parameter.requires_grad) 
        print('\n====================================================================')                        
    '''       
    print('\n====================================================================')
    print(f'    Epoch {epoch}/{args.epochs}')
    print('====================================================================')
    # Train on training set
    # Now training only with pairs.
    training_set_results__train_all = train_epoch(model, data_loader_training_set__train_all, len(df_training_set__train_all), log_vars, args.gradient_accumulation_steps, optimizer, scheduler, device, get_fixed_predictions=bool(strtobool(args.show_fixed_predictions)))
    number_training_instances = len(training_set_results__train_all['pred_labels'][task])        
    print(f'==== Training instances: {number_training_instances}')
    if train_annotators_train:
        print(f'==== Training annotators: {train_annotators_train}')        
    for task in tasks:
        print(f'\n========================= EVALUATION TASK: {task} ================================')      
        if bool(strtobool(args.eval_train_training)):        
            if config[task]['single_sent']:
                training_set_results__eval = predict(model, data_loader_training_set__eval_single, log_vars, device, get_fixed_predictions=bool(strtobool(args.show_fixed_predictions)))
            else:
                training_set_results__eval = predict(model, data_loader_training_set__eval_pairs, log_vars, device, get_fixed_predictions=bool(strtobool(args.show_fixed_predictions)))

            training_set_metrics = get_metrics(training_set_results__eval['pred_labels'][task], training_set_results__eval['real_labels'][task])
            number_eval_instances = len(training_set_results__eval['pred_labels'][task])       
            show_results('Training', training_set_results__eval, task, training_set_metrics, annotators=','.join(train_annotators_eval[task]), show_fixed_predictions=bool(strtobool(args.show_fixed_predictions)), epoch=epoch, num_instances=number_eval_instances)
        else:
            print('====================================================================')
            print(f'  Training set - Epoch: {epoch} - Annotators: {train_annotators_train} - {number_training_instances} instances')
            print('====================================================================')
            print(f'  Training Loss: {training_set_results__train_all["loss"]} - Task: {task}')
            
        # Predict / evaluate on test set
        if bool(strtobool(args.eval_test_training)):
            if config[task]['single_sent']:
                test_set_results__eval = predict(model, data_loader_test_set__eval_single, log_vars, device, get_fixed_predictions=bool(strtobool(args.show_fixed_predictions)))                
            else:
                test_set_results__eval = predict(model, data_loader_test_set__eval_pairs, log_vars, device, get_fixed_predictions=bool(strtobool(args.show_fixed_predictions)))                              

            test_set_metrics = get_metrics(test_set_results__eval['pred_labels'][task], test_set_results__eval['real_labels'][task]) 
            number_eval_instances = len(test_set_results__eval['pred_labels'][task])                
            show_results('Test', test_set_results__eval, task, test_set_metrics, annotators=','.join(test_annotators_eval[task]), show_fixed_predictions=bool(strtobool(args.show_fixed_predictions)), epoch=epoch, num_instances=number_eval_instances)

        # Save model in every epoch      
        if bool(strtobool(args.save_all_epochs)) and task in output_path_all_epochs:
            output_path_model = output_path_all_epochs[task] + '/epoch_' + str(epoch)
            if not os.path.exists(output_path_model):
                os.makedirs(output_path_model)
            print(f'Epoch: {epoch} - Saving model to {output_path_model}')
            model.save_model(save_directory=output_path_model)      
            tokenizer.save_pretrained(save_directory=output_path_model)
            
        # Save model in last epoch                
        elif bool(strtobool(args.save_last_epoch)) and task in output_path_last and epoch == args.epochs:
            print(f'Saving last model to {output_path_last[task]}')
            model.save_model(save_directory=output_path_last[task])      
            tokenizer.save_pretrained(save_directory=output_path_last[task])        

print('\n--- Training finished ---\n')




