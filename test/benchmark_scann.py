import scann
import numpy as np
import os
import sys
from dataclasses import dataclass, field
from typing import List
import time
import struct
import argparse
import subprocess

class Dataset:
  def __init__(self, dimension, num):
    self._dimension = dimension
    self._num = num
    self._labels = np.random.randint(0, 10, num)
    self._vecs = np.random.rand(num, dimension)
  
  def read_dataset(self, file_name):
      with open(file_name, 'rb') as file:
          # Calculate the size of each entry (1 uint32 + self._dimension float32s)
          entry_size = 1 * np.dtype(np.uint32).itemsize + self._dimension * np.dtype(np.float32).itemsize
          
          # Read the entire file into a numpy array of bytes
          data = np.frombuffer(file.read(), dtype=np.uint8)
          
          # Ensure that the data size is consistent with the expected size of the entries
          if data.size % entry_size != 0:
              raise ValueError("The file size does not match the expected entry size.")
          
          # Reshape the data to separate each entry
          reshaped_data = data.reshape(-1, entry_size)
          
          # Extract labels (first 4 bytes as uint32)
          labels = reshaped_data[:, :4].view(np.uint32).flatten()
          
          # Extract vectors (remaining bytes as float32)
          vectors = reshaped_data[:, 4:].view(np.float32)
      
      self._labels = labels
      self._vecs = vectors
  
@dataclass
class Query:
    label: int
    vec: List[float] = field(default_factory=list)
    
class QuerySet:
    def __init__(self, dimension: int = 0, num: int = 0):
        self._dimension = dimension
        self._num = num
        self._queries: List[Query] = []

    def read_queries(self, file_name):
        with open(file_name, 'rb') as file:
            # Calculate the size of each entry: 1 uint32 (label) and self._dimension float32s (vector)
            entry_size = 1 * np.dtype(np.uint32).itemsize + self._dimension * np.dtype(np.float32).itemsize
            
            # Read the entire file into a byte stream
            data = np.frombuffer(file.read(), dtype=np.uint8)
            
            # Check if data size is consistent with the expected entry size
            if data.size % entry_size != 0:
                raise ValueError("File size does not match the expected entry size.")
            
            # Reshape data to separate each entry
            reshaped_data = data.reshape(-1, entry_size)
            
            # Extract labels (first 4 bytes as uint32)
            labels = reshaped_data[:, :4].view(np.uint32).flatten()
            
            # Extract vectors (remaining bytes as float32)
            vectors = reshaped_data[:, 4:].view(np.float32)
        
        # Create a list of Query objects
        self._queries = [Query(label, vec) for label, vec in zip(labels, vectors)]
        
def build_scann(dataset, top_k, num_leaves, num_leaves_to_search):
    build_start = time.time()
    searcher = scann.scann_ops_pybind.builder(dataset, top_k, "dot_product").tree(
        num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search, training_sample_size=250000).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(top_k).build()
    build_end = time.time()
    build_time = build_end - build_start
    print("Build Time(s):", build_time, flush=True)
    return searcher

def run_scann(searcher, query_vector, top_k, base):
  print("Base:", base, flush=True)
  search_start = time.time()
  neighbors, distances = searcher.search_batched_parallel(query_vector, final_num_neighbors=top_k, 
                                                        leaves_to_search=base, pre_reorder_num_neighbors=top_k*10)
  search_end = time.time()
  
  average_search_time = (search_end - search_start) / len(query_vector) * 1000
  print("QPS: ", 1000 / average_search_time, flush=True)
  return neighbors
  
def save_vector_result(file_name: str, results: list[list[int]], top_k: int):
    with open(file_name, 'wb') as file:
        for result in results:
            file.write(struct.pack('I', top_k))
            file.write(struct.pack(f'{top_k}I', *result[:top_k]))

def save_label_result(file_name: str, results: list[list[int]], top_k: int):
    with open(file_name, 'wb') as file:
        for result in results:
            file.write(struct.pack('I', top_k))
            file.write(struct.pack(f'{top_k}I', *result[:top_k]))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Calculate recall for a given dataset and algorithm.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("dataset", type=str, help="Dataset name")
    parser.add_argument("train_dataset_name", type=str, help="Train dataset name")
    parser.add_argument("test_dataset_name", type=str, help="Test dataset name")
    parser.add_argument("algorithm", type=str, help="Algorithm name")
    parser.add_argument("top_k", type=int, help="Top K value for recall calculation")
    parser.add_argument("data_num", type=int, help="Data num")
    parser.add_argument("query_num", type=int, help="Query num")
    parser.add_argument("dim", type=int, help='dimension')
    parser.add_argument("num_leaves", type=int, help="Num leaves")
    parser.add_argument("num_leaves_to_search", type=int, help='search param')
    parser.add_argument("data_type", type=str, help='data type')
    parser.add_argument("mode", type=str, help='mode')
    parser.add_argument("index_path", type=str, help='index path')
    # New argument for the project's root path
    parser.add_argument("--project_root", type=str, required=True, help="Root directory of the benchmark project")
    
    args = parser.parse_args()
    
    source_path = os.path.join(args.dataset_path, f"{args.train_dataset_name}.bin")
    query_path = os.path.join(args.dataset_path, f"{args.test_dataset_name}.bin")

    dataset = Dataset(args.dim, args.data_num)
    queryset = QuerySet(args.dim, args.query_num)

    dataset.read_dataset(source_path)
    queryset.read_queries(query_path)

    searcher = build_scann(dataset._vecs, args.top_k, args.num_leaves, args.num_leaves_to_search)
    
    # Construct paths from the project_root argument
    recall_script_path = os.path.join(args.project_root, 'tools', f'recall_{args.data_type}.py')
    benchmark_index_path = os.path.join(args.project_root, 'benchmark_index')
    
    command = [
        'python', recall_script_path,
        args.dataset_path,
        args.dataset,
        args.train_dataset_name,
        args.test_dataset_name,
        args.algorithm,
        str(args.top_k),
        'ip',
        args.dataset,
        benchmark_index_path
    ]
    
    base_group = [1, 2, 5, 8, 12, 16, 20, 30, 40, 50, 60, 80, 100, 120, 200, 300, 500, 1000, 2000, 3000]
    
    # Create the main result directory
    result_dir = os.path.join(benchmark_index_path, args.algorithm)
    os.makedirs(result_dir, exist_ok=True)
    
    for base in base_group:
        neighbors = run_scann(searcher, [query.vec for query in queryset._queries], args.top_k, base)
        
        result_path = os.path.join(result_dir, f"{args.dataset}.result")
        label_result_path = os.path.join(result_dir, f"{args.dataset}.result_top{args.top_k}.label")
        
        save_vector_result(result_path, neighbors, args.top_k)
        res_label = [dataset._labels[x] for x in neighbors]
        save_label_result(label_result_path, [dataset._labels[x] for x in neighbors], args.top_k)
        
        subprocess.run(command)