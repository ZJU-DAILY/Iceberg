#!/usr/bin/env bash

# --- Configuration ---
# Set the project's root directory here.
PROJECT_ROOT="/path/to/your/project"

# --- Source Dataset Configuration ---
source "${PROJECT_ROOT}/scripts/config_dataset.sh"
dataset_imagenet1k_avg


# --- Build ---
if [ ! -d "build" ]; then
  echo "Directory build does not exist, creating it"
  mkdir build
fi

cd build
cmake ..
make -j8

# --- Parameters & Paths ---
pre_path="${PROJECT_ROOT}/benchmark_index"
algorithm="napg"
mode="search"
efc=256
# efs=(100 150 200 250 300 350 400 450 500 550 600 800 1000 1500)
efs=(100 200 300 400 500 600 800 1000 1500)
M=32
norm_range=100
sample_size=10
type="ip"

INDEX_PREFIX_PATH="${pre_path}/${algorithm}/${PREFIX}_M${M}_L${efc}.index"
RESULT_PREFIX_PATH="${pre_path}/${algorithm}/${PREFIX}_M${M}_L${efc}.result"
log_file="${pre_path}/${algorithm}/${PREFIX}_M${M}_L${efc}.log"
recall_path="${PROJECT_ROOT}/tools/recall_${DATASET_TYPE}.py"
result_name="${PREFIX}_M${M}_L${efc}"

# --- Execution ---
case "$mode" in
  build)
    echo "Building index..."
    ./test/benchmark_napg ${BASE_PATH} ${QUERY_FILE} ${mode} $DATA_DIM $K ${efc} ${M} ${norm_range} ${sample_size} ${INDEX_PREFIX_PATH} ${RESULT_PREFIX_PATH}  | tee -a "$log_file"
    ;;
  search)
    echo "Searching index..."
    for ef_search in "${efs[@]}"; do
      echo "========================================" | tee -a "$log_file"
      echo "Running with efs: $ef_search" | tee -a "$log_file"
      ./test/benchmark_napg ${BASE_PATH} ${QUERY_FILE} ${mode} ${DATA_DIM} ${K} ${efc} ${M} ${norm_range} ${sample_size} ${INDEX_PREFIX_PATH} ${RESULT_PREFIX_PATH} ${ef_search} | tee -a "$log_file"
      python3 ${recall_path} ${DATA_PRE_PATH} ${PREFIX} ${TRAIN_NAME} ${TEST_NAME} ${algorithm} ${K} ${type} ${result_name} ${pre_path}| tee -a "$log_file"
      echo "========================================" | tee -a "$log_file"
    done
    ;;
  *)
    echo "Invalid mode. Use 'build' or 'search'."
    exit 1
    ;;
esac