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

# TRAINING-TEST SETS
TRAINING_SET_NAME=training
TEST_SET_NAME=test-consensus
EVAL_TEST_TRAINING=True

TRAINING_SET_PATH=../data/${DOMAIN}/pairs_both/${TRAINING_SET_NAME}.tsv
TEST_SET_PATH=../data/${DOMAIN}/pairs_one/${TEST_SET_NAME}.tsv

ADDITIONAL_FEATURES=adu1_pos:POS        
ENCODING_SINGLE=same
DROPOUT=0.2
SAVE_ALL_MODELS=True    
SAVE_LAST_MODEL=False
FREEZE_BERT=False

for TASK in rel_direction adu_type main_unit adu_afu
do      
    BASE_DIR_MODELS=${MODELS_PATH}/${DOMAIN}/stl/$NOW/${TRAINING_SET_NAME}_${TEST_SET_NAME}/${TASK}
    BASE_DIR_LOGS=${LOGS_PATH}/${DOMAIN}/stl/$NOW/${TRAINING_SET_NAME}_${TEST_SET_NAME}/${TASK}
    if [[ ! -d ${BASE_DIR_LOGS} ]] ; then        
        echo "STL ${DOMAIN} - ${TASK}"
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
            --task=${TASK} \
            --dropout=${DROPOUT} \
            --encoding_single=${ENCODING_SINGLE} \
            --save_all_epochs=${SAVE_ALL_MODELS} \
            --save_last_epoch=${SAVE_LAST_MODEL} \
            --eval_test_training=${EVAL_TEST_TRAINING} \
            --freeze_bert=${FREEZE_BERT}
        echo 
    else
        echo "Skipping STL ${DOMAIN} - ${TASK}: ${BASE_DIR_LOGS} exists"
    fi                 
done

echo
echo `date` " --- FINISHED ALL PROCESSES ---"
echo

