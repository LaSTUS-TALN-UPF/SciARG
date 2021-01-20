#!/bin/bash

PYTHON=python3
SCRIPT=../train_mtl.py
MODEL_NAME_OR_PATH=allenai/scibert_scivocab_cased

NOW=$(date +"%Y%m%d")

#DOMAIN=cl
DOMAIN=bio
# Set to background if the script is to be run in background. In this case, also set the logs path.
MODE=background
GRAD_ACC=2
EPOCHS=25

DATA_PATH=../../data
MODELS_PATH=../../models
LOGS_PATH=../../logs

# TRAINING-TEST SETS
TRAINING_SET_NAME=training
TEST_SET_NAME=test-consensus
EVAL_TEST_TRAINING=True

TRAINING_SET_PATH=${DATA_PATH}/${DOMAIN}/pairs_both/${TRAINING_SET_NAME}.tsv
TEST_SET_PATH=${DATA_PATH}/${DOMAIN}/pairs_one/${TEST_SET_NAME}.tsv

ADDITIONAL_FEATURES=adu1_pos:POS
ENCODING_SINGLE=same
SAVE_ALL_MODELS=True    
SAVE_LAST_MODEL=False
    
BASE_DIR_MODELS=${MODELS_PATH}/${DOMAIN}/mtl/$NOW/${TRAINING_SET_NAME}_${TEST_SET_NAME}
BASE_DIR_LOGS=${LOGS_PATH}/${DOMAIN}/mtl/$NOW/${TRAINING_SET_NAME}_${TEST_SET_NAME}
if [[ ! -d ${BASE_DIR_LOGS} ]] ; then        
    echo `date` " - MTL ${DOMAIN} - " ${TRAINING_SET_NAME} ${TEST_SET_NAME}
    ${PYTHON} -u ${SCRIPT} \
        --domain=${DOMAIN} \
        --epochs=${EPOCHS} \
        --gradient_accumulation_steps=${GRAD_ACC} \
        --additional_features=${ADDITIONAL_FEATURES} \
        --training_set_path=${TRAINING_SET_PATH} \
        --test_set_path=${TEST_SET_PATH} \
        --mode=${MODE} \
        --base_dir_logs=${BASE_DIR_LOGS} \
        --base_dir_models=${BASE_DIR_MODELS} \
        --model_name_or_path=${MODEL_NAME_OR_PATH} \
        --encoding_single=${ENCODING_SINGLE} \
        --save_all_epochs=${SAVE_ALL_MODELS} \
        --save_last_epoch=${SAVE_LAST_MODEL} \
        --eval_test_training=${EVAL_TEST_TRAINING}
    echo 
else
    echo "Skipping MTL ${DOMAIN}: ${BASE_DIR_LOGS} exists"
fi                 

echo
echo `date` " --- FINISHED ALL PROCESSES ---"
echo

