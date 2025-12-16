#include <iostream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include "ip-nsw+/hnswlib.h"
#include <algorithm>
#include <ctime>
#include <cassert>
#include "io.h"
#include <gtest/gtest.h>


int main(int argc, char** argv) {
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
    u_int32_t efConstruction = std::stoi(argv[6]);
    u_int32_t cos_efConstruction = std::stoi(argv[7]);
    u_int32_t M = std::stoi(argv[8]);
    u_int32_t cos_M = std::stoi(argv[9]);
    u_int32_t cos_efsearch = std::stoi(argv[10]);
    std::string store_index_path = std::string(argv[11]);
    std::string result_path =  std::string(argv[12]);
    benchmark::DataSet dataset;
    benchmark::QuerySet queryset;
    benchmark::IO io;

    io.ReadDataSet(source_path, dataset, dimension);
    io.ReadQuerySet(query_path, queryset, dimension);
    std::cout << "Dataset size: " << dataset._num << std::endl;
    std::cout << "Queryset size: " << queryset._num << std::endl;

    if (mode == "build") {
      hnswlib::InnerProductSpace space(dimension);
      std::vector<float> element_norms;
      element_norms.reserve(dataset._num);
      for (size_t i = 0; i < dataset._num; ++i) {
          float line_norm = 0;
          for (size_t j = 0; j < dimension; ++j) {
              float ele = dataset._vecs[i][j];
              line_norm += ele * ele;
          }
          line_norm = sqrt(line_norm);
          element_norms.push_back(line_norm);
      }
      hnswlib::HierarchicalNSW<float>* alg_hnsw =
          new hnswlib::HierarchicalNSW<float>(&space, dataset._num, M, cos_M,
                                              efConstruction, cos_efConstruction);
      alg_hnsw->elementNorms = std::move(element_norms);
      auto s_insert = std::chrono::system_clock::now();
      #pragma omp parallel for
      for (size_t i = 0; i < dataset._num; ++i) {
        alg_hnsw->addPoint(dataset._vecs[i].data(), i);
      }

      alg_hnsw->saveIndex(store_index_path);
      auto e_insert = std::chrono::system_clock::now();
      std::chrono::duration<double> insert_time = e_insert - s_insert;
      std::cout << "Insert time: " << insert_time.count() << std::endl;
    }else {
      u_int32_t efSearch = std::stoi(argv[13]);
      hnswlib::InnerProductSpace space(dimension);
      hnswlib::HierarchicalNSW<float>* alg_hnsw =
          new hnswlib::HierarchicalNSW<float>(&space, store_index_path, false);
      std::vector<std::vector<std::pair<size_t, float>>> result(queryset._num);
      std::vector<std::vector<u_int32_t>> knns(queryset._num);
      std::vector<std::vector<u_int32_t>> knns_label(queryset._num);
      int *temp_data1 = NULL;
      int *temp_data2 = NULL;
      int degree_count_ip = 0;
      int degree_count_cos = 0;
      float norm_avg = 0, norm_var = 0;
      for (int i = 0; i < alg_hnsw->maxelements_; ++i) {
          temp_data1 = (int *)(alg_hnsw->data_level0_memory_ + i * alg_hnsw->size_data_per_element_);
          temp_data2 = (int *)(alg_hnsw->data_level0_memory_ + i * alg_hnsw->size_data_per_element_ + alg_hnsw->size_links_level0_ip_);
          degree_count_ip += *temp_data1;
          degree_count_cos += *temp_data2;
          norm_avg += alg_hnsw->elementNorms[i];
      }
      norm_avg /= alg_hnsw->maxelements_;
      for (int i = 0; i < alg_hnsw->maxelements_; ++i) {
          norm_var += (alg_hnsw->elementNorms[i] - norm_avg) * (alg_hnsw->elementNorms[i] - norm_avg);
      }
      norm_var /= alg_hnsw->maxelements_;
      alg_hnsw->setEf(efSearch);
      alg_hnsw->setCosEf(cos_efsearch);
      auto s_search = std::chrono::system_clock::now();
  #pragma omp parallel for
      for (size_t i = 0; i < queryset._num; ++i) {
        auto query = queryset._queries[i];
        auto query_vec = query._vec;
        auto query_label = query._label;
        auto result = alg_hnsw->searchKnn(query_vec.data(), top_k);
        while (!result.empty()) {
          auto top = result.top();
          knns[i].push_back(top.second);
          knns_label[i].push_back(dataset._label[top.second]);
          result.pop();
        }
        std::reverse(knns[i].begin(), knns[i].end());
        std::reverse(knns_label[i].begin(), knns_label[i].end());
      }
      auto e_search = std::chrono::system_clock::now();
      std::chrono::duration<double> search_time = e_search - s_search;
      std::cout << "search time: " << search_time.count() / queryset._num * 1000
      << "ms" << std::endl;
      std::cout << "query per second: " << 1000.0 / (search_time.count() / queryset._num * 1000) << " qps" << std::endl;
      io.SaveVectorResult(result_path, knns, top_k);
      io.SaveLabelResult(result_path + "_top" + std::to_string(top_k) + ".label", knns_label, top_k);
    }
    return 0;
}