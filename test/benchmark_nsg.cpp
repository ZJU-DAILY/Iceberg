#include <cstring>

#include "io.h"
#include "nsg/util.h"
#include "efanna2e/index_random.h"
#include "efanna2e/util.h"
#include "efanna2e/index_graph.h"
#include "nsg/index_nsg.h"


float* get_data(std::vector<std::vector<float>> &vecs, u_int32_t num, u_int32_t dim) {
    float *data = new float[1ll * num * dim];
    size_t index = 0;
    for (int i = 0; i < num; i++) {
        memcpy(data + 1ll * i * dim, vecs[i].data(), dim * sizeof(float));
    }
    return data;
}

void build_knng(unsigned points_num, unsigned dim, float* data_load, const char* graph_filename,
    unsigned K, unsigned L, unsigned iter, unsigned S, unsigned R) {
    efanna2e::IndexRandom init_index(dim, points_num);
    efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index*)(&init_index));

    efanna2e::Parameters paras;
    paras.Set<unsigned>("K", K);
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("iter", iter);
    paras.Set<unsigned>("S", S);
    paras.Set<unsigned>("R", R);

    auto s = std::chrono::high_resolution_clock::now();
    std::cout << "Building knng\n";
    index.Build(points_num, data_load, paras);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e-s;
    std::cout <<"Build knng time cost: "<< diff.count() << "\n";

    index.Save(graph_filename);
}

void build_nsg(unsigned points_num, unsigned dim, float* data_load, const char* knng_path, const char *index_path,
                unsigned L, unsigned R, unsigned C) {
    efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);
    
    std::cout << "Building nsg index...\n";
    auto s = std::chrono::high_resolution_clock::now();
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("C", C);
    paras.Set<std::string>("nn_graph_path", knng_path);
    index.Build(points_num, data_load, paras);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    std::cout << "indexing time: " << diff.count() << "\n";
    index.Save(index_path);
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

    unsigned knng_L = (unsigned)atoi(argv[6]);
    unsigned knng_iter = (unsigned)atoi(argv[7]);
    unsigned knng_S = (unsigned)atoi(argv[8]);
    unsigned knng_R = (unsigned)atoi(argv[9]);

    unsigned build_L = (unsigned)atoi(argv[10]);
    unsigned build_R = (unsigned)atoi(argv[11]);
    unsigned build_C = (unsigned)atoi(argv[12]);

    std::string store_index_path = std::string(argv[13]);
    std::string knng_path = std::string(argv[14]);
    std::string result_path = std::string(argv[15]);

    std::cout << "Index Path: " << store_index_path << '\n';
    std::cout << "Knng path: " << knng_path << '\n';

    benchmark::DataSet dataset;
    benchmark::QuerySet queryset;
    benchmark::IO io;
    io.ReadDataSet(source_path, dataset, dimension);
    io.ReadQuerySet(query_path, queryset, dimension);
    std::cout << "Dataset size: " << dataset._num << std::endl;
    std::cout << "Queryset size: " << queryset._num << std::endl;

    float *data_load = get_data(dataset._vecs, dataset._num, dataset._dimension);
    dataset._vecs.clear();

    if (mode == "build") {       
        build_knng(dataset._num, dataset._dimension, data_load, knng_path.c_str(),
        top_k, knng_L, knng_iter, knng_S, knng_R); 
        build_nsg(dataset._num, dataset._dimension, data_load, knng_path.c_str(), store_index_path.c_str(),
                    build_L, build_R, build_C);
    } else {
        u_int32_t efSearch = std::stoi(argv[16]);
        efanna2e::IndexNSG index(dataset._dimension, dataset._num, efanna2e::L2, nullptr);
        index.Load(store_index_path.c_str());

        unsigned L = (unsigned)efSearch;
        unsigned K = (unsigned)top_k;

        if (L < K) {
            std::cout << "efSearch cannot be smaller than top_k!" << std::endl;
            exit(-1);
        }

        efanna2e::Parameters paras;
        paras.Set<unsigned>("L_search", L);
        paras.Set<unsigned>("P_search", L);

        std::vector<std::vector<u_int32_t>> knns(queryset._num);
        std::vector<std::vector<u_int32_t>> knns_label(queryset._num);

        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (size_t i = 0; i < queryset._num; i++) {
            std::vector<unsigned> tmp(K);
            auto query = queryset._queries[i];
            auto query_vec = query._vec;
            index.Search(query_vec.data(), data_load, K, paras, tmp.data());
            
            for (int j = 0; j < K; j++) {
                knns[i].push_back(tmp[j]);
                knns_label[i].push_back(dataset._label[tmp[j]]);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::chrono::duration<double> search_time = end - start;
        std::cout << "search time: " << search_time.count() / queryset._num * 1000
              << "ms" << std::endl;
        std::cout << "query per second: " << 1000.0 / (search_time.count() / queryset._num * 1000) << " qps" << std::endl;

        io.SaveVectorResult(result_path, knns, top_k);
        io.SaveLabelResult(result_path + "_top" + std::to_string(top_k) + ".label", knns_label, top_k);
    }

    delete[] data_load;
    return 0;
}