// #include <gtest/gtest.h>
#define EIGEN_DONT_PARALLELIZE
#define USE_AVX2
#include <omp.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <malloc.h>
#define FAST_SCAN
#include "io.h"

size_t DIM = 128;
size_t numC = 1;
size_t BB = 128;
size_t B_QUERY = 4;
size_t nprobe = 1;

#include "rabitq/src/matrix.h"
#include "rabitq/src/utils.h"
#include "rabitq/src/ivf_rabitq.h"
#include <getopt.h>

void print_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);

    std::cout << "Memory usage: " << usage.ru_maxrss / 1024 / 1024 << " GB" << std::endl;
}

int main(int argc, char** argv) {
    // --- Argument Parsing ---
    // Note: A project_root path is now required as the last argument.
    if (argc < 12) {
        std::cerr << "Usage: " << argv[0] 
                  << " [source_path] [query_path] [mode] [dimension] [top_k] "
                  << "[efConstruction] [M] [store_index_path] [result_path] "
                  << "[dataset_name] [project_root] <efSearch_if_search_mode>" 
                  << std::endl;
        return 1;
    }

    std::string source_path = std::string(argv[1]);
    std::string query_path = std::string(argv[2]);
    std::string mode = std::string(argv[3]);
    size_t dimension = std::stoi(argv[4]);
    size_t top_k = std::stoi(argv[5]);
    size_t efConstruction = std::stoi(argv[6]);
    size_t M = std::stoi(argv[7]);
    std::string store_index_path = std::string(argv[8]);
    std::string result_path = std::string(argv[9]);
    std::string dataset_name = std::string(argv[10]);
    std::string project_root = std::string(argv[11]); // New argument for the project's root path

    DIM = dimension;
    numC = efConstruction;
    BB = M;

    // --- Dataset Loading ---
    benchmark::DataSet dataset;
    benchmark::QuerySet queryset;
    benchmark::IO io;
    io.ReadDataSet(source_path, dataset, dimension);
    io.ReadQuerySet(query_path, queryset, dimension);

    std::cout << "Dataset size: " << dataset._num << std::endl;
    std::cout << "Queryset size: " << queryset._num << std::endl;

    // --- Path Configuration ---
    std::string store_data_path = project_root + "/benchmark_index/rabitq/";
    std::string python_script_base_path = project_root + "/baselines/rabitq/data/";

    if (mode == "build") {
        std::cout << "[RaBitQ] Building index\n";
        // Preprocess Data by calling external python scripts
        auto s_insert = std::chrono::system_clock::now();
        std::string shell;
        shell = "python " + python_script_base_path + "ivf.py " + source_path + " " + std::to_string(DIM) + " " + store_data_path + " " + dataset_name + " " + std::to_string(numC);
        system(shell.c_str());
        shell = "python " + python_script_base_path + "rabitq.py " + source_path + " " + std::to_string(DIM) + " " + store_data_path + " " + dataset_name + " " + std::to_string(numC) + " " + std::to_string(BB);
        system(shell.c_str());
        auto e_insert = std::chrono::system_clock::now();
        std::chrono::duration<double> insert_time = e_insert - s_insert;

        // Load Preprocessed Data
        char centroid_path[256], x0_path[256], dist_to_centroid_path[256], cluster_id_path[256], binary_path[256];

        Matrix<float> X(dataset._num, dataset._dimension);
        for(size_t i = 0; i < dataset._num; i++){
            memcpy(X.data + i * dataset._dimension, dataset._vecs[i].data(), sizeof(float) * dataset._dimension);
        }

        snprintf(centroid_path, sizeof(centroid_path), "%s%s_RandCentroid_C%d_B%d.fvecs", store_data_path.c_str(), dataset_name.c_str(), numC, BB);
        Matrix<float> C(centroid_path);

        snprintf(x0_path, sizeof(x0_path), "%s%s_x0_C%d_B%d.fvecs", store_data_path.c_str(), dataset_name.c_str(), numC, BB);
        Matrix<float> x0(x0_path);

        snprintf(dist_to_centroid_path, sizeof(dist_to_centroid_path), "%s%s_dist_to_centroid_%d.fvecs", store_data_path.c_str(), dataset_name.c_str(), numC);
        Matrix<float> dist_to_centroid(dist_to_centroid_path);

        snprintf(cluster_id_path, sizeof(cluster_id_path), "%s%s_cluster_id_%d.ivecs", store_data_path.c_str(), dataset_name.c_str(), numC);
        Matrix<uint32_t> cluster_id(cluster_id_path);

        snprintf(binary_path, sizeof(binary_path), "%s%s_RandNet_C%d_B%d.Ivecs", store_data_path.c_str(), dataset_name.c_str(), numC, BB);
        Matrix<uint64_t> binary(binary_path);

        // Build and Save Index
        s_insert = std::chrono::system_clock::now();
        IVFRN ivf(DIM, BB, X, C, dist_to_centroid, x0, cluster_id, binary);
        ivf.save(&store_index_path[0]);
        e_insert = std::chrono::system_clock::now();
        insert_time += e_insert - s_insert;
        std::cout << "Insert time: " << insert_time.count() << std::endl;

    } else { // Search mode
        if (argc < 13) {
            std::cerr << "Error: efSearch argument is required for search mode." << std::endl;
            return 1;
        }
        size_t efSearch = std::stoi(argv[12]);
        nprobe = efSearch;

        Matrix<float> Q(queryset._num, queryset._dimension);
        for(size_t i = 0; i < queryset._num; i++){
            memcpy(Q.data + i * queryset._dimension, queryset._queries[i]._vec.data(), sizeof(float) * queryset._dimension);
        }

        char transformation_path[256] = "";
        snprintf(transformation_path, sizeof(transformation_path), "%s%s_P_C%d_B%d.fvecs", store_data_path.c_str(), dataset_name.c_str(), numC, BB);
        Matrix<float> P(transformation_path);
        
        // Load index and search
        IVFRN ivf(DIM, BB);
        ivf.load(&store_index_path[0]);

        Matrix<float> RandQ(Q.n, BB, Q);
        RandQ = mul(RandQ, P);
        
        std::vector<std::vector<u_int32_t>> knns(queryset._num);
        std::vector<std::vector<u_int32_t>> knns_label(queryset._num);
        
        auto s_search = std::chrono::system_clock::now();
        #pragma omp parallel for
        for (size_t i = 0; i < Q.n; ++i) {
            ResultHeap result = ivf.search(Q.data + i * Q.d, RandQ.data + i * RandQ.d, top_k, nprobe);
            while (!result.empty()) {
                int id = result.top().second;
                result.pop();
                knns[i].push_back(id);
                knns_label[i].push_back(dataset._label[id]);
            }
            std::reverse(knns[i].begin(), knns[i].end());
            std::reverse(knns_label[i].begin(), knns_label[i].end());
            while(knns[i].size() < top_k) {
                knns[i].push_back(0);
            }
        }
        auto e_search = std::chrono::system_clock::now();
        
        std::chrono::duration<double> search_time = e_search - s_search;
        std::cout << "Search time: " << search_time.count() / queryset._num * 1000 << "ms" << std::endl;
        std::cout << "QPS: " << queryset._num / search_time.count() << " qps" << std::endl;
        
        io.SaveVectorResult(result_path, knns, top_k);
        io.SaveLabelResult(result_path + "_top" + std::to_string(top_k) + ".label", knns_label, top_k);
    }
    return 0;
}