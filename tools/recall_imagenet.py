import numpy
import os
import sys
import struct
from collections import defaultdict
import argparse

def read_results(filename):
    result = []
    with open(filename, 'rb') as f:
        while True:
            GK_bytes = f.read(4)
            if not GK_bytes:
                break  
            GK = struct.unpack('I', GK_bytes)[0]  
            result_bytes = f.read(4 * GK)
            result_1 = list(struct.unpack(f'{GK}I', result_bytes))
            result.append(result_1)
    return result

def read_label_results(filename):
    result = []
    with open(filename,'rb') as f:
        while True:
            GK_bytes = f.read(4)
            if not GK_bytes:
                break
            GK = struct.unpack('I', GK_bytes)[0]  
            result_bytes = f.read(4 * GK)
            result_1 = list(struct.unpack(f'{GK}I', result_bytes))
            result.append(result_1)
    return result

def read_query_labels(filename):
    labels = []
    with open(filename, 'rb') as f:
        while True:
            label_bytes = f.read(4)
            if not label_bytes:
                break
            label = struct.unpack('I', label_bytes)[0]
            labels.append(label)
    return labels

def read_groundtruth(filename):
    gt = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split()
            gt.append(line)
    return gt

def recall(gt, res, top_k):
    recall = 0
    for i in range(len(res)):
        res_i = res[i][0:top_k]
        gt_i = gt[i]
        gt_i = set([int(x) for x in gt_i])
        res_i = set(res_i)
        recall += len(gt_i & res_i)
        
    return recall / (len(res) * top_k)

def recall_label(gt_label, res_label, base_label_map, top_k):
    recall_search = 0
    recall_base = 0
    for i in range(len(res_label)):
        res_i = res_label[i]
        gt_i = int(gt_label[i])
        recall_search += res_i.count(gt_i)
        if base_label_map[gt_i] > top_k:
            recall_base += top_k
        else:
            recall_base += base_label_map[gt_i]

    return recall_search / recall_base
         

def label_mapping(base_label_file):
    label_map = defaultdict(int)
    with open(base_label_file, 'rb') as f:
        while True:
            label_bytes = f.read(4)
            if not label_bytes:
                break
            label = struct.unpack('I', label_bytes)[0]
            label_map[label] += 1
    return label_map
 
def knn_major_classification(res_label, gt_label, top_k):
    recall = 0
    for i in range(len(res_label)):
        res_i = res_label[i]
        # print(len(res_i))
        gt_i = int(gt_label[i])
        res_i = res_i[:top_k]
        res_i = [int(x) for x in res_i]
        res_i = numpy.array(res_i)
        res_i = numpy.bincount(res_i)
        res_i = numpy.argmax(res_i)
        # print(res_i)
        if res_i == gt_i:
            recall += 1
    
    print(f"Majority Classification Recall@{top_k}:\t", recall / len(res_label))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate recall for a given dataset and algorithm.")
    parser.add_argument("DATA_PRE_PATH", type=str, help="Path to the dataset")
    parser.add_argument("PREFIX", type=str, help="Dataset name")
    parser.add_argument("train_dataset_name", type=str, help="Train dataset name")
    parser.add_argument("test_dataset_name", type=str, help="Test dataset name")
    parser.add_argument("algorithm", type=str, help="Algorithm name")
    parser.add_argument("top_k", type=int, help="Top K value for recall calculation")
    parser.add_argument("type", type=str, help="Type of search")
    parser.add_argument("full_name", type=str, help="Full name of the dataset")
    parser.add_argument("pre_path", type=str, help="Pre path for the index")

    args = parser.parse_args()

    dataset_path = args.DATA_PRE_PATH
    dataset_name = args.PREFIX
    train_dataset_name = args.train_dataset_name
    test_dataset_name = args.test_dataset_name
    algorithm = args.algorithm
    top_k = args.top_k
    search_type = args.type
    full_name = args.full_name
    pre_index_path = args.pre_path
    
    
    gt_file = f"{dataset_path}/result-{dataset_name}-{search_type}-top100.txt"
    res_file = f"{pre_index_path}/{algorithm}/{full_name}.result"
    
    res_label_file = f"{pre_index_path}/{algorithm}/{full_name}.result_top{top_k}.label"
    
    query_label_file = f"{dataset_path}/{test_dataset_name}-label.bin"
    base_label_file = f"{dataset_path}/{train_dataset_name}-label.bin"
    gt = read_groundtruth(gt_file)
    res = read_results(res_file)  
    res_label = read_label_results(res_label_file)

    # # print(res_label)
    gt_label = read_query_labels(query_label_file)
    base_label_map = label_mapping(base_label_file)
    label_acc = recall_label(gt_label, res_label, base_label_map, top_k)
    # print(res_label)
    acc = recall(gt, res, top_k)

    print("Recall@100: ", acc)
    print("Label Recall@100: ", label_acc)

    for k in (5, 10, 50, 100):
      knn_major_classification(res_label, gt_label, k)
