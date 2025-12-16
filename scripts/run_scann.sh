#!/bin/bash

# --- Configuration ---
# Set the project's root directory here.
PROJECT_ROOT="/path/to/your/project"

# --- Source Dataset Configuration ---
source "${PROJECT_ROOT}/scripts/config_dataset.sh"
dataset_imagenet1k_avg

# --- Parameters & Paths ---
pre_path="${PROJECT_ROOT}/benchmark_index"
algorithm="scann"
num_leaves=2000
num_leaves_to_search=1000
mode="build"
INDEX_PREFIX_PATH="${pre_path}/${algorithm}/${PREFIX}.index"

# --- Execution ---
python3 "${PROJECT_ROOT}/test/benchmark_scann.py" \
  ${DATA_PRE_PATH} \
  ${PREFIX} \
  ${TRAIN_NAME} \
  ${TEST_NAME} \
  ${algorithm} \
  ${K} \
  ${data_num} \
  ${query_num} \
  ${DATA_DIM} \
  ${num_leaves} \
  ${num_leaves_to_search} \
  ${DATASET_TYPE} \
  ${mode} \
  ${INDEX_PREFIX_PATH} | tee -a "${pre_path}/${algorithm}/${PREFIX}.log"