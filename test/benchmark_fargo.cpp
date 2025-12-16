#include <string.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "fargo/Preprocess.h"
#include "fargo/mf_alsh.h"
#include "io.h"

int main(int argc, char** argv) {
  std::string source_path = "";
  std::string query_path = "";

  std::string mode = "";
  std::string full_dataset_name = "";

  u_int32_t dimension = 300;
  u_int32_t top_k = 100;
  u_int32_t L = 5;
  u_int32_t K = 12;
  u_int32_t ub = 100;
  float c = 0.8f;
  float break_ratio = 1.0;

  if (argc > 1) {
    source_path = std::string(argv[1]);
    query_path = std::string(argv[2]);
    mode = std::string(argv[3]);
    dimension = std::stoi(argv[4]);
    top_k = std::stoi(argv[5]);
    L = std::stoi(argv[6]);
    K = std::stoi(argv[7]);
    ub = std::stoi(argv[8]);
    c = std::stof(argv[9]);
    break_ratio = std::stof(argv[10]);
  }

  std::string store_index_path = std::string(argv[11]);
  std::string result_path = std::string(argv[12]);

  benchmark::DataSet dataset;
  benchmark::QuerySet queryset;
  benchmark::IO io;
  io.ReadDataSet(source_path, dataset, dimension);
  io.ReadQuerySet(query_path, queryset, dimension);
  // queryset._num = 500;
  std::cout << "Dataset size: " << dataset._num << std::endl;
  std::cout << "Queryset size: " << queryset._num << std::endl;
  // queryset._num = 10000;
  fargo::InnerProductSpace space(dimension);
  fargo::Preprocess prep(dataset, queryset, &space, top_k);
  if (mode == "build") {
    auto s_insert = std::chrono::system_clock::now();
    fargo::mf_alsh::Hash myslsh(prep, L, K, c);
    myslsh.saveIndex(store_index_path);
    auto e_insert = std::chrono::system_clock::now();
    std::chrono::duration<double> insert_time = e_insert - s_insert;
    std::cout << "Insert time: " << insert_time.count() << "s" << std::endl;
  } else if (mode == "search") {
    fargo::mf_alsh::Hash myslsh(store_index_path);
    auto s_search = std::chrono::system_clock::now();
    auto result =
        fargo::Alg0_mfalsh(myslsh, c, ub + top_k, top_k, L, K, prep, ub, break_ratio);
    std::vector<std::vector<u_int32_t>> result_label(queryset._num);
    for (int i = 0; i < queryset._num; i++) {
      for (auto x : result[i]) {
        result_label[i].push_back(dataset._label[x]);
      }
    }
    auto e_search = std::chrono::system_clock::now();
    std::chrono::duration<double> search_time = e_search - s_search;
    std::cout << "search time: " << search_time.count() / queryset._num * 1000
              << "ms" << std::endl;
    std::cout << "query per second: " << 1000.0 / (search_time.count() / queryset._num * 1000) << " qps" << std::endl;
    io.SaveVectorResult(result_path, result, top_k);
    io.SaveLabelResult(result_path + "_top" + std::to_string(top_k) + ".label", result_label, top_k);
  } else {
    std::cerr << "Please enter build or search." << std::endl;
  }

  return 0;
}