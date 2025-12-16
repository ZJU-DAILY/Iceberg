#include <vamana/index.h>
#include <string.h>
#include "vamana/utils.h"
#include <sys/mman.h>
#include <unistd.h>
#include "vamana/memory_mapper.h"
#include <random>
#include <iomanip>
#include <type_traits>
#include <iostream>
#include <boost/program_options.hpp>
#include <thread>
#include "sys/stat.h"
#include "io.h"
using namespace std;

int build_in_memory_index(const std::string&  data_path, const unsigned num_data, const unsigned dim,
    const diskann::Metric& metric, const unsigned R, const unsigned L, const unsigned C, const float alpha,
    const std::string& save_path) {
    diskann::Parameters paras;
    const unsigned num_threads = omp_get_num_threads();
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>(
    "C", C);  // maximum candidate set size during pruning procedure
    paras.Set<float>("alpha", alpha);
    paras.Set<bool>("saturate_graph", 0);
    paras.Set<unsigned>("num_threads", num_threads);
    auto                  s = std::chrono::high_resolution_clock::now();
    diskann::Index<float> index(metric, data_path.c_str(), num_data, dim);
    index.build(paras);
    std::chrono::duration<double> diff =
    std::chrono::high_resolution_clock::now() - s;

    std::cout << "Indexing time: " << diff.count() << " \npath index: "<< save_path<< "\n";
    s = std::chrono::high_resolution_clock::now();
    index.save((save_path).c_str());
    diff =
    std::chrono::high_resolution_clock::now() - s;
    std::cout << "Saving time: " << diff.count()<<" \n";
    return 0;
}


int search_memory_index(string data_file, string result_path, unsigned num_data, string memory_index_file, size_t dim,
  string query_bin, size_t num_query, _u64 L, _u64 K, benchmark::DataSet & dataset) {
    float* query = nullptr;

    size_t              query_num, query_dim, query_aligned_dim;

    diskann::Metric metric;

    metric = diskann::Metric::L2;

    _u64              recall_at = K;


    diskann::load_data<float>(query_bin, query, num_query, dim, query_aligned_dim);
    diskann::load_aligned_bin<float>(query_bin, query, num_query, dim,
                                     query_aligned_dim);

    query_num=num_query;
    query_dim=dim;

    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);

    diskann::Index<float> index(metric, data_file.c_str(), num_data,dim);
    index.load(memory_index_file.c_str());
    auto mx = 0;
    long long avg = 0;
    auto mi = std::numeric_limits<int>::max();
    vector<unsigned> out_deg(index._final_graph.size());
    for (auto i = 0 ; i < index._final_graph.size(); i ++) {
        out_deg[i] = index._final_graph[i].size();
    }
    for (auto i = 0 ; i < index._final_graph.size(); i ++) {
        avg += out_deg[i];
        if (out_deg[i] > mx) {
            mx = out_deg[i];
        }
        if (out_deg[i] < mi) {
            mi = out_deg[i];
        }
    }
    std::vector<unsigned> ratios;
    for (auto i = 0 ; i < index._final_graph.size(); i ++) {
        auto ratio = 1.0 * index._final_graph[i].size() / out_deg[i];
        ratios.push_back(index._final_graph[i].size());
    }

    std::cout << "avg node: " << 1.0 * avg / index._final_graph.size() << std::endl;
    std::cout << "max node: " << mx << std::endl;
    std::cout << "min node: " << mi << std::endl;

    diskann::Parameters paras;

    std::vector<std::vector<uint32_t>> knns(query_num);
    std::vector<std::vector<u_int32_t>> knns_label(query_num);
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int64_t i = 0; i < (int64_t) query_num; i++) {
        std::vector<unsigned> tmp(K);
        index.search(query + i * query_aligned_dim, recall_at, L,
        tmp.data());
        for (int j = 0; j < recall_at; j++) {
            knns[i].push_back(tmp[j]);
            knns_label[i].push_back(dataset._label[knns[i][j]]);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::chrono::duration<double> search_time = end - start;
    std::cout << "search time: " << search_time.count() / query_num * 1000
          << "ms" << std::endl;
    std::cout << "query per second: " << 1000.0 / (search_time.count() / query_num * 1000) << " qps" << std::endl;
    benchmark::IO io;
    io.SaveVectorResult(result_path, knns, recall_at);
    io.SaveLabelResult(result_path + "_top" + std::to_string(recall_at) + ".label", knns_label, recall_at);
    diskann::aligned_free(query);
    return 0;
}


int main(int argc, char **argv) {
    std::string source_path = "";
    std::string query_path = "";
    std::string mode = "";

    if (argc > 1) {
        source_path = std::string(argv[1]);
        query_path = std::string(argv[2]);
        mode = std::string(argv[3]);
    }
    u_int32_t dimension = std::stoi(argv[4]);
    u_int32_t top_k = std::stoi(argv[5]);

    unsigned L = (unsigned)atoi(argv[6]);
    unsigned R = (unsigned)atoi(argv[7]);
    unsigned C = (unsigned)atoi(argv[8]);
    float alpha = (float)std::atof(argv[9]);
    std::string store_index_path = std::string(argv[10]);
    std::string result_path = std::string(argv[11]);

    std::cout << "Index Path: " << store_index_path << '\n';

    benchmark::DataSet dataset;
    benchmark::QuerySet queryset;
    benchmark::IO io;
    io.ReadDataSet(source_path, dataset, dimension);
    io.ReadQuerySet(query_path, queryset, dimension);
    std::cout << "Dataset size: " << dataset._num << std::endl;
    std::cout << "Queryset size: " << queryset._num << std::endl;


    if (mode == "build") {
        auto metric = diskann::Metric::L2;
        build_in_memory_index(source_path, dataset._num, dataset._dimension, metric, R, L, C, alpha, store_index_path);
    }
    else {
        u_int32_t efSearch = std::stoi(argv[12]);
        unsigned L = (unsigned)efSearch;
        unsigned K = (unsigned)top_k;

        if (L < K) {
            std::cout << "efSearch cannot be smaller than top_k!" << std::endl;
            exit(-1);
        }

        search_memory_index(source_path, result_path, dataset._num, store_index_path, queryset._dimension, query_path, queryset._num, L, K,  dataset);
    }

    return 0;
}