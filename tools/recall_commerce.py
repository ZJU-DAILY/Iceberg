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
            labels.append(int(label))
    return labels

def read_groundtruth(filename):
    gt = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
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

def parse_user_interact_items(items_str):
    items = []
    for item in items_str.split(';'):
        if item:
            item_id, label = item.split(':')
            items.append((int(item_id), int(label)))
    return items

def parse_hot_items(hot_items_str):
    hot_items = []
    for item in hot_items_str.split(';'):
        if item:
            item_id, hot_levels = item.split(':')
            hot_level1, hot_level2 = map(float, hot_levels.split(','))
            hot_items.append((int(item_id), hot_level1, hot_level2))
            # hot_items[int(item_id)] = [hot_level1, hot_level2]
    return hot_items

def extract_query_data(file_path):
    query_data = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            q_id, q_type, user_interact_items, hot_items = line.split('|')
            
            q_id = int(q_id)
            q_type = int(q_type)
            user_interact_items = parse_user_interact_items(user_interact_items)
            hot_items = parse_hot_items(hot_items)
            
            query_data[q_id] = {
                'q_type': q_type,
                'user_interact_items': user_interact_items,
                'hot_items': hot_items
            }
    
    return query_data

def recall_hot(query_data, query_label, res, base_label, top_k, hot_label_type):
    recall_hot = 0
    hot_tot = 0
    
    for i in range(len(res)):
        res_i = res[i][:top_k]
        res_i = [int(base_label[x]) for x in res_i]
        hot_items = query_data[query_label[i]]['hot_items']
        hot_items_set = {item_id for item_id, _, _ in hot_items}
        hot_item_hot_level = {item_id: (hot_level1, hot_level2) for item_id, hot_level1, hot_level2 in hot_items}
        hot_tot += sum(hot_item_hot_level[item_id][hot_label_type] for item_id in hot_items_set)
        recall_hot += sum(hot_item_hot_level[item_id][hot_label_type] for item_id in res_i if item_id in hot_items_set)
        
    return recall_hot, recall_hot / hot_tot if hot_tot != 0 else 0


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

    query_type = 'high'

    gt_file = f"{dataset_path}/result-{dataset_name}-{query_type}-{search_type}-top100.txt"
    res_file = f"{pre_index_path}/{algorithm}/{full_name}.result"
    gt = read_groundtruth(gt_file)
    res = read_results(res_file)
    acc = recall(gt, res, top_k)

    query_label_file = f"{dataset_path}/{test_dataset_name}_label.bin"
    base_label_file = f"{dataset_path}/{train_dataset_name}_label.bin"
    meta_file = f"{dataset_path}/q_{query_type}_meta.txt"

    query_label = read_query_labels(query_label_file)
    base_label = read_query_labels(base_label_file)
    meta_query_data = extract_query_data(meta_file)
    pop_score, _ = recall_hot(meta_query_data, query_label, res, base_label, top_k, 1)
    
    print("Recall@100: ", acc)
    print(f"Popularity Score@100: ", pop_score)