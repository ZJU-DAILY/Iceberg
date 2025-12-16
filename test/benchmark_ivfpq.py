import faiss
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
        self._labels = np.random.randint(0, 10, num, dtype=np.int64)
        self._vecs = np.random.rand(num, dimension).astype(np.float32)
    
    def read_dataset(self, file_name):
        print(f"Reading dataset from {file_name}...")
        with open(file_name, 'rb') as file:
            entry_size = 1 * np.dtype(np.uint32).itemsize + self._dimension * np.dtype(np.float32).itemsize
            data = np.frombuffer(file.read(), dtype=np.uint8)
            reshaped_data = data.reshape(-1, entry_size)
            self._labels = reshaped_data[:, :4].view(np.uint32).flatten().astype(np.int64)
            self._vecs = reshaped_data[:, 4:].view(np.float32)
        print(f"Dataset loaded. Shape: {self._vecs.shape}")

@dataclass
class Query:
    label: int
    vec: List[float] = field(default_factory=list)

class QuerySet:
    def __init__(self, dimension: int = 0):
        self._dimension = dimension
        self._queries: List[Query] = []

    def read_queries(self, file_name, num_queries: int):
        print(f"Reading queries from {file_name}...")
        with open(file_name, 'rb') as file:
            entry_size = 1 * np.dtype(np.uint32).itemsize + self._dimension * np.dtype(np.float32).itemsize
            data = np.fromfile(file, dtype=np.uint8, count=num_queries * entry_size)
            reshaped_data = data.reshape(-1, entry_size)
            labels = reshaped_data[:, :4].view(np.uint32).flatten()
            vectors = reshaped_data[:, 4:].view(np.float32)
        self._queries = [Query(label, vec.tolist()) for label, vec in zip(labels, vectors)]
        print(f"{len(self._queries)} queries loaded.")

def build_ivfpq(dataset: Dataset, nlist: int, pqm: int, index_file_name: str):
    d = dataset._dimension
    xb = dataset._vecs.astype(np.float32)
    xt = xb[:1000000] if xb.shape[0] > 1000000 else xb
    full_index_path = index_file_name
    
    if os.path.exists(full_index_path):
        print(f"Index file already exists at {full_index_path}. Skipping build.")
        return
    
    print(f"Building new index: IVF{nlist},PQ{pqm}x8")
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, pqm, 8)
    begin_time = time.time()
    faiss.omp_set_num_threads(48)
    print(f"Training index with {faiss.omp_get_max_threads()} threads...")
    index.train(xt)
    print("Adding vectors to index...")
    index.add(xb)
    end_time = time.time()
    print(f"Index built in {end_time - begin_time:.2f} seconds.")
    print(f"Saving index to {full_index_path}")
    faiss.write_index(index, full_index_path)

def save_results(file_name: str, results: np.ndarray):
    output_dir = os.path.dirname(file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    results_u32 = results.astype(np.uint32)

    with open(file_name, 'wb') as file:
        top_k = results_u32.shape[1]
        for result_row in results_u32:
            file.write(struct.pack('I', top_k))
            file.write(struct.pack(f'{top_k}I', *result_row))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run Faiss IVFPQ+Rerank, save results, and trigger recall calculation.")
    
    # File and dataset parameters
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument("dataset_name", type=str, help="Dataset name (e.g., 'sift')")
    parser.add_argument("train_dataset_name", type=str, help="Train dataset filename (e.g., 'sift_base.bin')")
    parser.add_argument("test_dataset_name", type=str, help="Test dataset filename (e.g., 'sift_query.bin')")
    parser.add_argument("recall_script_path", type=str, help="Path to the external recall calculation script")
    
    # Core parameters
    parser.add_argument("--dim", type=int, required=True, help='Dimension of vectors')
    parser.add_argument("--data_num", type=int, required=True, help="Number of vectors in the base dataset")
    parser.add_argument("--query_num", type=int, required=True, help="Number of query vectors")
    
    # Faiss parameters
    parser.add_argument("--algorithm", type=str, default="ivfpq_rerank", help="Algorithm name for result paths")
    parser.add_argument("--nlist", type=int, default=1024, help="Number of inverted lists (Voronoi cells)")
    parser.add_argument("--pqm", type=int, default=16, help='Number of sub-quantizers for PQ')
    
    # Search and evaluation parameters
    parser.add_argument("--top_k", type=int, default=100, help="Final top K neighbors for recall calculation")
    parser.add_argument("--rerank_k", type=int, default=200, help="Number of initial candidates to retrieve for reranking")
    parser.add_argument("--mode", type=str, choices=['build', 'search'], required=True, help='Mode: build index or search')
    
    # Path parameters
    parser.add_argument("--index_path", type=str, default='./faiss_indices', help='Directory to save/load the index')
    parser.add_argument("--output_path", type=str, default='./results', help="Base directory to save search results")
    parser.add_argument("--data_type", type=str, default="uint8", help='Data type for the recall script')

    args = parser.parse_args()
    
    source_path = f"{args.dataset_path}/{args.train_dataset_name}.bin"
    query_path = f"{args.dataset_path}/{args.test_dataset_name}.bin"
    
    index_name_str = f"IVF{args.nlist}_PQ{args.pqm}x8"
    index_file = f"{index_name_str}.index"
    full_index_path = args.index_path

    if args.mode == 'build':
        dataset = Dataset(args.dim, args.data_num)
        dataset.read_dataset(source_path)
        build_ivfpq(dataset, args.nlist, args.pqm, full_index_path)
        print("\nIndex build finished. You can now run in 'search' mode.")
        sys.exit(0)

    if not os.path.exists(full_index_path):
        print(f"Error: Index file not found at {full_index_path}")
        print("Please run with --mode build first.")
        sys.exit(1)

    dataset = Dataset(args.dim, args.data_num)
    dataset.read_dataset(source_path)

    queryset = QuerySet(args.dim)
    queryset.read_queries(query_path, args.query_num)

    print(f"Loading index from {full_index_path}...")
    search_index = faiss.read_index(full_index_path)
    print("Index loaded.")

    num_threads_for_search = 48
    faiss.omp_set_num_threads(num_threads_for_search)
    print(f"** Faiss search threads set to {faiss.omp_get_max_threads()} **")

    xq = np.array([q.vec for q in queryset._queries]).astype('float32')

    nprobe_group = [8, 16, 32, 64, 128, 256] 

    result_base_dir = os.path.join(args.output_path, args.algorithm)
    result_file_for_recall = os.path.join(result_base_dir, f"{args.dataset_name}.result")
    label_result_file_for_recall = os.path.join(result_base_dir, f"{args.dataset_name}.result_top{args.top_k}.label")

    for nprobe in nprobe_group:
        if nprobe > args.nlist:
            continue
        
        print(f"\n{'='*20} Testing nprobe = {nprobe} {'='*20}")
        search_index.nprobe = nprobe
        print(f"Searching for {args.rerank_k} candidates...")
        start_time = time.time()
        D_initial, I_initial = search_index.search(xq, args.rerank_k)
        end_time = time.time()
        qps = args.query_num / (end_time - start_time)
        print(f"Search and Rerank completed. QPS: {qps:.2f}",  flush=True)
        print(f"Saving results to {result_file_for_recall}...",  flush=True)
        
        final_indices = np.full((args.query_num, args.top_k), -1, dtype=np.int64)
        for i in range(args.query_num):
            candidate_ids = I_initial[i][I_initial[i] != -1]
            if len(candidate_ids) == 0: continue
            
            candidate_vectors = dataset._vecs[candidate_ids]
            exact_distances = np.linalg.norm(candidate_vectors - xq[i:i+1], axis=1)
            reranked_local_indices = np.argsort(exact_distances)
            
            num_final = min(args.top_k, len(reranked_local_indices))
            final_indices[i, :num_final] = candidate_ids[reranked_local_indices[:num_final]]
        
        save_results(result_file_for_recall, final_indices)
        
        final_labels = np.full_like(final_indices, -1, dtype=np.int64)
        valid_mask = final_indices != -1
        final_labels[valid_mask] = dataset._labels[final_indices[valid_mask]]
        
        print(f"Saving label results to {label_result_file_for_recall}...")
        save_results(label_result_file_for_recall, final_labels)

        print("Triggering external recall script...")
        command = [
            'python', args.recall_script_path,
            args.dataset_path,
            args.dataset_name,
            args.train_dataset_name,
            args.test_dataset_name,
            args.algorithm,
            str(args.top_k),
            'nn', 
            args.dataset_name,
            args.output_path,
            '--dataset-type', args.data_type
        ]
        
        print(f"Executing command: {' '.join(command)}")
        subprocess.run(command)