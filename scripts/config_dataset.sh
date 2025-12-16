#!/usr/bin/env bash

dataset_imagenet1k_dinov2() {
  BASE_PATH="/path/to/your/data/imagenet-1k/dinov2-train.bin"
  QUERY_FILE="/path/to/your/data/imagenet-1k/dinov2-validation.bin"
  PREFIX="dinov2"
  K=100
  DATA_DIM=768
  DATASET_TYPE="imagenet"
  DATA_PRE_PATH="/path/to/your/data/imagenet-1k"
  TRAIN_NAME="dinov2-train"
  TEST_NAME="dinov2-validation"
  data_num=1281167
  query_num=50000
}


dataset_imagenet1k_eva02() {
  BASE_PATH="/path/to/your/data/imagenet-1k/eva02-train.bin"
  QUERY_FILE="/path/to/your/data/imagenet-1k/eva02-validation.bin"
  PREFIX="eva02"
  K=100
  DATA_DIM=1024
  DATASET_TYPE="imagenet"
  DATA_PRE_PATH="/path/to/your/data/imagenet-1k"
  TRAIN_NAME="eva02-train"
  TEST_NAME="eva02-validation"
  data_num=1281167
  query_num=50000
}

dataset_imagenet1k_avg() {
  BASE_PATH="/path/to/your/data/imagenet-1k/convnext-avg-pool-train.bin"
  QUERY_FILE="/path/to/your/data/imagenet-1k/convnext-avg-pool-validation.bin"
  PREFIX="convnext-avg-pool"
  K=100
  DATA_DIM=1536
  DATASET_TYPE="imagenet"
  DATA_PRE_PATH="/path/to/your/data/imagenet-1k"
  TRAIN_NAME="convnext-avg-pool-train"
  TEST_NAME="convnext-avg-pool-validation"
  data_num=1281167
  query_num=50000
}

dataset_glink_ir101() {
  BASE_PATH="/path/to/your/data/glink/ir101-train.bin"
  QUERY_FILE="/path/to/your/data/glink/ir101-validation.bin"
  PREFIX="ir101"
  K=100
  DATA_DIM=512
  DATASET_TYPE="glink"
  DATA_PRE_PATH="/path/to/your/data/glink"
  TRAIN_NAME="ir101-train"
  TEST_NAME="ir101-validation"
  data_num=17091649
  query_num=20000
}

dataset_glink_vit() {
  BASE_PATH="/path/to/your/data/glink/vit-train.bin"
  QUERY_FILE="/path/to/your/data/glink/vit-validation.bin"
  PREFIX="vit"
  K=100
  DATA_DIM=512
  DATASET_TYPE="glink"
  DATA_PRE_PATH="/path/to/your/data/glink"
  TRAIN_NAME="vit-train"
  TEST_NAME="vit-validation"
  data_num=17091649
  query_num=20000
}

dataset_commerce() {
  BASE_PATH="/path/to/your/data/shopee_v1/commerce-train.bin"
  QUERY_FILE="/path/to/your/data/shopee_v1/commerce-validation_high.bin"
  PREFIX="commerce"
  K=100
  DATA_DIM=48
  DATASET_TYPE="commerce"
  DATA_PRE_PATH="/path/to/your/data/commerce"
  TRAIN_NAME="commerce-train"
  TEST_NAME="commerce-validation"
  data_num=99085171
  query_num=23893
}

dataset_bookcorpus() {
  BASE_PATH="/path/to/your/data/book_corpus/stella-train.bin"
  QUERY_FILE="/path/to/your/data/book_corpus/stella-validation.bin"
  PREFIX="book_corpus"
  K=100
  DATA_DIM=1024
  DATASET_TYPE="book"
  DATA_PRE_PATH="/path/to/your/data/book_corpus"
  TRAIN_NAME="stella-train"
  TEST_NAME="stella-validation"  
}