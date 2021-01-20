#!/bin/bash

PYTHON=python3
SCRIPT=../eval_mtl.py
MODE=
DATA_PATH=../../data
MODELS_PATH=../../models

ENCODING_SINGLE=same
ADDITIONAL_FEATURES=adu1_pos:POS        

for DOMAIN_EVAL in cl bio
do
    TEST_SET_PATH=${DATA_PATH}/${DOMAIN_EVAL}/pairs_one/test-consensus.tsv

    for MODEL_TYPE in mtl stl
    do
        for TASK in adu_type main_unit rel_direction adu_afu
        do
            for EPOCH in 5 10 15 20
            do
                if [[ "${MODEL_TYPE}" == "stl" ]]; then 
                    MODEL_NAME=model-${MODEL_TYPE}-${DOMAIN}-${TASK}-ep_${EPOCH}
                else
                    MODEL_NAME=model-${MODEL_TYPE}-${DOMAIN}-ep_${EPOCH}
                fi 
                MODEL_NAME_OR_PATH=${MODELS_PATH}/${MODEL_NAME}
                echo
                echo "EVAL - ${MODEL_NAME} in ${DOMAIN_EVAL} - Epoch ${EPOCH}"
                ${PYTHON} -u ${SCRIPT} \
                    --task=${TASK} \
                    --domain=${DOMAIN} \
                    --model_name_or_path=${MODEL_NAME_OR_PATH} \
                    --additional_features=${ADDITIONAL_FEATURES} \
                    --test_set_path=${TEST_SET_PATH} \
                    --encoding_single=${ENCODING_SINGLE} \
                    --show_confusion_matrix=True \
                    --mode=${MODE}
                echo
            done
        done
    done

done



echo `date` " --- FINISHED ALL PROCESSES ---"
echo

