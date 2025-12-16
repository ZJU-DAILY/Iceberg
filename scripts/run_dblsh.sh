#!/usr/bin/env bash

# Set the project's root directory here
PROJECT_ROOT="/path/to/your/project"

source "${PROJECT_ROOT}/scripts/config_dataset.sh"
dataset_imagenet1k_avg

if [ ! -d "build" ]; then
  echo "Directory build does not exist, creating it"
  mkdir build
fi

cd build
cmake ..
make -j8

# All paths below are now relative to the project root
pre_path="${PROJECT_ROOT}/benchmark_index"
algorithm="dblsh"
L=5
K_dblsh=10
type="nn"

INDEX_PREFIX_PATH="${pre_path}/${algorithm}/${PREFIX}_L${L}_K${K_dblsh}.index"
RESULT_PREFIX_PATH="${pre_path}/${algorithm}/${PREFIX}_L${L}_K${K_dblsh}.result"
log_file="${pre_path}/${algorithm}/${PREFIX}_L${L}_K${K_dblsh}.log"
recall_path="${PROJECT_ROOT}/tools/recall_${DATASET_TYPE}.py"
result_name="${PREFIX}_L${L}_K${K_dblsh}"

if [ $? -eq 0 ]; then
  beta=(0.8 0.9 0.92 0.94 0.96 0.98)
  for beta in "${beta[@]}"; do
    echo "========================================" | tee -a "$log_file"
    echo "Running with beta: $beta" | tee -a "$log_file"
    ./test/benchmark_dblsh ${BASE_PATH} ${QUERY_FILE} ${DATA_DIM} ${K} ${L} ${K_dblsh} 1.5 "$beta" 0.1 ${RESULT_PREFIX_PATH}  | tee -a "$log_file"
    python3 ${recall_path} ${DATA_PRE_PATH} ${PREFIX} ${TRAIN_NAME} ${TEST_NAME} ${algorithm} ${K} ${type} ${result_name} ${pre_path}| tee -a "$log_file"
    echo "========================================" | tee -a "$log_file"
  done
else
  echo "Build failed"
fi