#!/bin/bash

# --- Configuration ---
# Define the root directory of your project here.
PROJECT_ROOT="/path/to/your/project"

# Source the dataset configuration.
source "${PROJECT_ROOT}/scripts/config_dataset.sh"

# Select your dataset configuration function as needed.
dataset_imagenet1k_avg

# --- Environment Settings ---
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Parameters & Paths ---
pre_path="${PROJECT_ROOT}/benchmark_index"
algorithm="ivfpq" # Algorithm name, used for creating directories

mode="search"

nlist=12000  # Number of IVF clusters
pqm=48       # Number of sub-quantizers for PQ

rerank_k=300 # Number of candidate vectors for reranking

RECALL_SCRIPT_PATH="${PROJECT_ROOT}/tools/recall_${DATASET_TYPE}.py"
PYTHON_SCRIPT_PATH="${PROJECT_ROOT}/test/benchmark_ivfpq.py"

# --- Directory and File Setup ---
INDEX_DIR="${pre_path}/${algorithm}"
INDEX_FILENAME="${PREFIX}_ivf${nlist}_pq${pqm}.index"
FULL_INDEX_PATH="${INDEX_DIR}/${INDEX_FILENAME}"

LOG_DIR="${pre_path}/${algorithm}"
# Note: The 'nprobe' variable used below is not defined in this script.
LOG_FILE="${LOG_DIR}/${PREFIX}_${mode}_nprobe${nprobe}.log"
mkdir -p ${LOG_DIR} # Ensure the log directory exists

# --- Execution ---
# Pass all configurations as arguments to the Python script.
python ${PYTHON_SCRIPT_PATH} \
    ${DATA_PRE_PATH} \
    ${PREFIX} \
    ${TRAIN_NAME} \
    ${TEST_NAME} \
    ${RECALL_SCRIPT_PATH} \
    --dim "${DATA_DIM}" \
    --data_num "${data_num}" \
    --query_num "${query_num}" \
    --mode "${mode}" \
    --algorithm "${algorithm}" \
    --top_k "${K}" \
    --nlist "${nlist}" \
    --pqm "${pqm}" \
    --rerank_k "${rerank_k}" \
    --index_path "${FULL_INDEX_PATH}" \
    --output_path "${pre_path}" \
    --data_type "${DATASET_TYPE}" \
    | tee -a "${LOG_FILE}"