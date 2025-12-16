#include <iostream>
#include <fstream>
#include <time.h>
#include "dblsh/Preprocess.h"
#include "dblsh/dblsh.h"
#include "dblsh/Query.h"
#include "io.h"

int main(int argc, char const* argv[]){
	std::string source_path = "";
	std::string query_path = "";

	std::string full_dataset_name = "";
	
	u_int32_t dimension = 300;
	u_int32_t top_k = 100;
	u_int32_t L = 5;
	u_int32_t K = 10;
	float c = 1.5;
	float beta = 0.3;
	float R_min = 0.1f;

	if (argc > 1) {
		source_path = std::string(argv[1]);
		query_path = std::string(argv[2]);
		dimension = std::stoi(argv[3]);
		top_k = std::stoi(argv[4]);
		L = std::stoi(argv[5]);
		K = std::stoi(argv[6]);
		c = std::stof(argv[7]);
		beta = std::atof(argv[8]);
		R_min = std::atof(argv[9]);
	}

  std::string result_path = std::string(argv[10]);

	benchmark::DataSet dataset;
	benchmark::QuerySet queryset;
	benchmark::IO io;
	io.ReadDataSet(source_path, dataset, dimension);
	io.ReadQuerySet(query_path, queryset, dimension);

	std::cout << "Dataset size: " << dataset._num << std::endl;
	std::cout << "Queryset size: " << queryset._num << std::endl;

	dblsh::L2Space space(dimension);
	dblsh::Preprocess prep(dataset, queryset, &space);
	dblsh::Parameter param(prep, L, K, R_min);
	
	auto s_insert = std::chrono::system_clock::now();
	dblsh::Hash myslsh(prep, param);
	auto e_insert = std::chrono::system_clock::now();
	std::chrono::duration<double> insert_time = e_insert - s_insert;
	std::cout << "Insert time: " << insert_time.count() << "s" << std::endl;

	auto s_search = std::chrono::system_clock::now();
	auto result = dblsh::lshknn(c, top_k, myslsh, prep, beta);
	std::cout << "here" << std::endl;
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
	return 0;
}
