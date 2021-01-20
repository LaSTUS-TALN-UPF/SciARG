from argparse import ArgumentParser
from datetime import datetime
import os

parser = ArgumentParser()

parser.add_argument('--domain', type=str, default='bio', help='Domain.')
args, unknown = parser.parse_known_args()  
if args.domain != 'bio' and args.domain != 'cl':
    args.domain = 'cl'    

# Default values
# MODEL_NAME_OR_PATH = 'bert-base-cased'
MODEL_NAME_OR_PATH = 'allenai/scibert_scivocab_cased'

# By default no task - tasks taken from config.py.   
TASK = ''   

# Annotators to filter - they are overwritten when the program is called.
ANNOTATORS = '' 
# Filter specific for training set  
TRAIN_ANNOTATORS = '' 
# Filter specific for val/test set    
TEST_ANNOTATORS = ''   
    
# These arguments can be overriden by task-specific parameters in config.py
LABEL_FIELD = 'aty'
BEST_METRIC = 'w_f1'

TEXT_FIELDS = 'adu1_text|adu2_text'
POSITION_FIELDS = 'adu1_pos|adu2_pos'
ADD_DISTANCE_TOKENS = 'True'
ADD_ORDER_TOKENS = 'True'
SPECIAL_TOKENS = '[ROOT]'        
#ADDITIONAL_FEATURES = 'doc_id:DOC,adu_pos:POS,annotator:ANNOTATOR'
ADDITIONAL_FEATURES = 'adu1_pos:pos'
INSTANCE_ID_FIELD_SINGLE = 'adu1_id'
INSTANCE_ID_FIELD_PAIR = 'rel_id'
TRAINING_SET_PATH = '../data/' + args.domain + '/mtl/pairs_both/training.tsv'
TEST_SET_PATH = '../data/' + args.domain + '/mtl/pairs_one/test-consensus.tsv'   
SAVE_PREDICTIONS_PATH = '../data/' + args.domain + '/mtl/pairs_one/test-predictions.tsv' 

#MODE = 'background'
MODE = 'interactive'
# Whether to evaluate on test set while training
EVAL_TEST_TRAINING = 'True'
EVAL_TRAIN_TRAINING = 'False'
NUM_CV_FOLDS = 0
CV_FOLD = 0
CV_SPLIT_TASK = 'rel_direction'
ENCODING_SINGLE = 'different' #'same' or 'different' as pairs.
SHUFFLE = 'True'
# This applies only when training on training set and evaluating on validation set. Otherwise, all epochs are saved.
FREEZE_BERT = 'False'
SAVE_ALL_EPOCHS = 'True'
SAVE_LAST_EPOCH = 'True'
BASE_DIR_MODELS = '../models/' + args.domain
BASE_DIR_LOGS = '../logs/' + args.domain
EPOCHS = 3
LR = 2e-5
RANDOM_SEED = 42
MAXLEN = 0
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_STEPS_PERCENTAGE = 0.1
DROPOUT = 0.1
NUM_WORKERS = 1
DATE_TIME =  datetime.now().strftime("%Y-%m-%d_%H%M")
DO_LOWER_CASE = 'False'
# Path to file with training/test/dev data, which is automatically splitted - if paths to previously splitted files are are not provided below. 
#DATASET_PATH_SPLIT = ''
DEFAULT_LABEL = 'proposal'
COLUMNS_OUTPUT = ''
POOLED_TOKENS = ''
SHOW_ANNOTATOR_FIELD = 'annotator'
# Show detailed predictions / confusion matrix when running evaluation
SHOW_PROBABILITIES = 'True'
SHOW_FIXED_PREDICTIONS = 'False'
SHOW_CONFUSION_MATRIX = 'True'

# Note: Boolean values always passed as strings to avoid problems when they are included in external calls (as 'False' evaluates to True).

# Add dropout
parser.add_argument('--model_name_or_path', type=str, default=MODEL_NAME_OR_PATH, help='Name of or path to the pretrained/trained model.')
parser.add_argument('--task', type=str, default=TASK, help='Task.')
parser.add_argument('--encoding_single', type=str, default=ENCODING_SINGLE, help='Use pair or single encodings for single-sentence tasks.')
parser.add_argument('--mode', type=str, default=MODE, help='Mode in which the program is called (interactive or background) - used to redirect output.')
parser.add_argument('--do_lower_case', type=str, default=DO_LOWER_CASE, help='Convert to lower-case when tokenizing.')
parser.add_argument('--num_cv_folds', type=int, default=NUM_CV_FOLDS, help='Number of cross-validations folds.')
parser.add_argument('--cv_fold', type=int, default=CV_FOLD, help='Cross-validation fold to process. It is assumed that the same random_state is used to get the same split in different calls.')
parser.add_argument('--cv_split_task', type=str, default=CV_SPLIT_TASK, help='Task used for stratified cross-validation split.')
parser.add_argument('--annotators', type=str, default=ANNOTATORS, help='All annotators to consider (default).')
parser.add_argument('--train_annotators', type=str, default=TRAIN_ANNOTATORS, help='Filter annotations by annotator(s) in training set.')
parser.add_argument('--test_annotators', type=str, default=TEST_ANNOTATORS, help='Filter annotations by annotator(s) in test set.')
parser.add_argument('--show_annotator_field', type=str, default=SHOW_ANNOTATOR_FIELD, help='Field with annotator. Leave empty if do not want to show it.')
parser.add_argument('--text_fields', type=str, default=TEXT_FIELDS, help='Column name of text field.')
parser.add_argument('--position_fields', type=str, default=POSITION_FIELDS, help='Column names of position fields.')
parser.add_argument('--add_distance_tokens', type=str, default=ADD_DISTANCE_TOKENS, help='Add tokens indicating absolute distance between the two texts?')
parser.add_argument('--add_order_tokens', type=str, default=ADD_ORDER_TOKENS, help='Add tokens indicating whether the first text occurs before or after the second one.')
parser.add_argument('--label_field', type=str, default=LABEL_FIELD, help='Column name of label field.')
parser.add_argument('--additional_features', type=str, default=ADDITIONAL_FEATURES, help='Column names of additional features to be added as special tokens. Format: <column_name>:<token_prefix_to_add_to_sent1>...|<column_name>:<token_prefix_to_add_to_sent2>...')
parser.add_argument('--special_tokens', type=str, default=SPECIAL_TOKENS, help='Comma-separated special tokens to add to tokenizer.')
parser.add_argument('--instance_id_field_pair', type=str, default=INSTANCE_ID_FIELD_PAIR, help='Column name to use as instance identifier for pairs of sentences.')
parser.add_argument('--instance_id_field_single', type=str, default=INSTANCE_ID_FIELD_SINGLE, help='Column name to use as instance identifier for one sentence.')
parser.add_argument('--maxlen', type=int, default=MAXLEN, help='Maximum number of tokens in the input sequence during training.')
parser.add_argument('--train_batch_size', type=int, default=TRAIN_BATCH_SIZE, help='Batch size during training.')
parser.add_argument('--eval_batch_size', type=int, default=EVAL_BATCH_SIZE, help='Batch size during evaluation.')
parser.add_argument('--gradient_accumulation_steps', type=int, default=GRADIENT_ACCUMULATION_STEPS, help='Gradient accumulation steps.')
parser.add_argument('--best_metric', type=str, default=BEST_METRIC, help='Metric to use to save best validation set model (acc, w_f1, mac_f1).')
parser.add_argument('--lr', type=float, default=LR, help='Learning rate for Adam.')
parser.add_argument('--warmup_steps_percentage', type=float, default=WARMUP_STEPS_PERCENTAGE, help='Percentage of warmup steps for learning rate, calculated based on epochs, batch and training set sizes.')
parser.add_argument('--dropout', type=float, default=DROPOUT, help='Dropout.')
parser.add_argument('--shuffle', type=str, default=SHUFFLE, help='Shuffle training data in each epoch.')
parser.add_argument('--freeze_bert', type=str, default=FREEZE_BERT, help='Freeze BERT parameters and train only heads.')
parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs.')
parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='Number of workers.')
#parser.add_argument('--dataset_path_split', type=str, default=DATASET_PATH_SPLIT, help='Dataset path (automatically splitted into train, test, dev).')
parser.add_argument('--random_seed', type=int, default=RANDOM_SEED, help='Random seed to split dataset.')
parser.add_argument('--training_set_path', type=str, default=TRAINING_SET_PATH, help='Training set path.')
parser.add_argument('--test_set_path', type=str, default=TEST_SET_PATH, help='Test set path.')
parser.add_argument('--save_predictions_path', type=str, default=SAVE_PREDICTIONS_PATH, help='Path to save predictions.')
parser.add_argument('--base_dir_logs', type=str, default=BASE_DIR_LOGS, help='Where to save the logs.')
parser.add_argument('--base_dir_models', type=str, default=BASE_DIR_MODELS, help='Where to save the models.')
parser.add_argument('--date_time', type=str, default=DATE_TIME, help='Date/time to add to the model name when saving.')
parser.add_argument('--eval_test_training', type=str, default=EVAL_TEST_TRAINING, help='Whether to evaluate on test set during training.')
parser.add_argument('--eval_train_training', type=str, default=EVAL_TRAIN_TRAINING, help='Whether to evaluate on training set during training.')
parser.add_argument('--save_all_epochs', type=str, default=SAVE_ALL_EPOCHS, help='Whether to save the model in every epoch. Otherwise the best validation models are saved.')
parser.add_argument('--save_last_epoch', type=str, default=SAVE_LAST_EPOCH, help='Whether to save the last model.')
parser.add_argument('--show_probabilities', type=str, default=SHOW_PROBABILITIES, help='Show probabilities when predicting labels.')
parser.add_argument('--show_fixed_predictions', type=str, default=SHOW_FIXED_PREDICTIONS, help='Fix predictions.')
parser.add_argument('--show_confusion_matrix', type=str, default=SHOW_CONFUSION_MATRIX, help='Show confusion matrix for test set.')
parser.add_argument('--default_label', type=str, default=DEFAULT_LABEL, help='Label ignored, only used to populate test set (for predictions).')
parser.add_argument('--columns_output', type=str, default=COLUMNS_OUTPUT, help='Additional columns to include in predictions (besides real and predicted labels).')
parser.add_argument('--pooled_tokens', type=str, default=POOLED_TOKENS, help='Position of the tokens to be included in the pooled representation of the text (in addition to 0, which is CLS, always included).')

args = parser.parse_args()


        
        

