#include "mag/index_mag.h"
#include "mag/util.h"
#include "io.h"

void save_results(std::string result_file, std::vector<std::vector<unsigned>> &kmips) {
  std::ofstream out(result_file);
  if (!out.is_open()) {
    std::cerr << "Cannot open result file: " << result_file << std::endl;
    exit(1);
  }
  for (unsigned i = 0; i < kmips.size(); i++) {
    for (unsigned j = 0; j < kmips[i].size(); j++) {
      if (j != kmips[i].size() - 1) {
        out << kmips[i][j] << " ";
      } else {
        out << kmips[i][j] << "\n";
      }
    }
  }
}

float* get_data(std::vector<std::vector<float>> &vecs, u_int32_t num, u_int32_t dim) {
  float *data = new float[1ll * num * dim];
  size_t index = 0;
  for (int i = 0; i < num; i++) {
      memcpy(data + 1ll * i * dim, vecs[i].data(), dim * sizeof(float));
  }
  return data;
}

int main(int argc, char** argv) {
  float* data_load = nullptr;

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
  unsigned C = (unsigned)atof(argv[8]);
  unsigned R_IP = (unsigned)atoi(argv[9]);
  unsigned M = (unsigned)atoi(argv[10]);
  unsigned threshold = (unsigned)atoi(argv[11]); 
  
  std::string store_index_path = std::string(argv[12]);
  std::string knng_path = std::string(argv[13]);
  std::string result_path = std::string(argv[14]);


  benchmark::DataSet dataset;
  benchmark::QuerySet queryset;
  benchmark::IO io;
  io.ReadDataSet(source_path, dataset, dimension);
  io.ReadQuerySet(query_path, queryset, dimension);
  std::cout << "Dataset size: " << dataset._num << std::endl;
  std::cout << "Queryset size: " << queryset._num << std::endl;


  data_load = get_data(dataset._vecs, dataset._num, dataset._dimension);
  dataset._vecs.clear();
  // data_load = MAG::data_align(data_load, dataset._num, dataset._dimension);
  omp_set_num_threads(48);

  if (mode == "build") {
    std::cout << "L = " << L << ", ";
    std::cout << "R = " << R << ", ";
    std::cout << "C = " << C << std::endl;
    std::cout << "R_IP = " << R_IP << std::endl;
    std::cout << "M = " << M << std::endl;
    MAG::IndexMAG index(dataset._dimension, dataset._num);
    MAG::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("C", C);
    paras.Set<unsigned>("R_IP", R_IP);
    paras.Set<unsigned>("threshold", threshold);
    paras.Set<unsigned>("n_try", 1);
    paras.Set<unsigned>("M", M);
    paras.Set<std::string>("nn_graph_path", knng_path);

    std::cout << "Output ARDG Path: " << argv[12] << std::endl;

    auto s = std::chrono::high_resolution_clock::now();
    index.Build(dataset._num, data_load, paras);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "Build Time: " << diff.count() << "\n";
    index.Save(store_index_path.c_str());
  } else {
      u_int32_t efSearch = std::stoi(argv[15]);
      MAG::IndexMAG index(dataset._dimension, dataset._num);
      index.Load(store_index_path.c_str());

      unsigned L = (unsigned)efSearch;
      unsigned K = (unsigned)top_k;
      std::cout << "L = " << L << ", ";
      std::cout << "K = " << K << std::endl;

      MAG::Parameters paras;
      paras.Set<unsigned>("L_search", L);

      std::vector<std::vector<u_int32_t>> knns(queryset._num);
      std::vector<std::vector<u_int32_t>> knns_label(queryset._num);

      index.entry_point_candidate(data_load);

      auto start = std::chrono::high_resolution_clock::now();
      #pragma omp parallel for
      for (size_t i = 0; i < queryset._num; i++) {
          std::vector<unsigned> tmp(K);
          auto query = queryset._queries[i];
          auto query_vec = query._vec;
          index.Search_NN_IP(query_vec.data(), data_load, K, paras, tmp.data());
          
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

  return 0;
}