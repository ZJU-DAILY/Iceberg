#pragma once

#include "common_includes.h"
#include "format.h"

namespace benchmark {
  class IO {
    public:
    std::string LoadDatasetPath(const std::string &dataset_name) {

    }
    void ReadDataSet(const std::string &file_name, DataSet &data_set, u_int32_t dimension) {
      std::ifstream file(file_name);
      if (!file.is_open()) {
        throw std::invalid_argument("Cannot open file: " + file_name);
      }
      // Calculate the size of each data entry (one int + 768 floats)
      size_t entry_size = sizeof(u_int32_t) + dimension * sizeof(float);

      // Reserve space for the dataset based on the file size
      file.seekg(0, std::ios::end);
      size_t file_size = file.tellg();
      size_t num_entries = file_size / entry_size;
      file.seekg(0, std::ios::beg);

      data_set.reserve(num_entries);
      data_set._dimension = dimension;
      data_set._num = num_entries;

      for (size_t i = 0; i < num_entries; ++i) {
          u_int32_t label;
          std::vector<float> vec(dimension);

          // Read the label (int)
          file.read(reinterpret_cast<char*>(&label), sizeof(u_int32_t));

          // Read the vector (768 floats)
          file.read(reinterpret_cast<char*>(vec.data()), dimension * sizeof(float));

          // Store in dataset
          data_set._label.push_back(label);
          data_set._vecs.push_back(vec);
      }
    }

    void ReadQuerySet(const std::string &file_name, QuerySet &query_set, u_int32_t dimension) {
      std::ifstream file(file_name);
      if (!file.is_open()) {
        throw std::invalid_argument("Cannot open file: " + file_name);
      }
      // Calculate the size of each query entry (one int + 768 floats)
      size_t entry_size = sizeof(u_int32_t) + dimension * sizeof(float);

      // Reserve space for the dataset based on the file size
      file.seekg(0, std::ios::end);
      size_t file_size = file.tellg();
      size_t num_entries = file_size / entry_size;
      file.seekg(0, std::ios::beg);

      query_set.reserve(num_entries);
      query_set._dimension = dimension;
      query_set._num = num_entries;

      for (size_t i = 0; i < num_entries; ++i) {
          u_int32_t label;
          std::vector<float> vec(dimension);

          // Read the label (int)
          file.read(reinterpret_cast<char*>(&label), sizeof(u_int32_t));

          // Read the vector (768 floats)
          file.read(reinterpret_cast<char*>(vec.data()), dimension * sizeof(float));

          // Store in dataset
          Query query;
          query._label = label;
          query._vec = vec;
          query_set._queries.push_back(query);
      }
    }

    void ReadPureDataSet(const std::string &file_name, PureDataSet &data_set, u_int32_t dimension) {
      std::ifstream file(file_name);
      if (!file.is_open()) {
        throw std::invalid_argument("Cannot open file: " + file_name);
      }
      // Calculate the size of each data entry (one int + 768 floats)
      size_t entry_size = dimension * sizeof(float);

      // Reserve space for the dataset based on the file size
      file.seekg(0, std::ios::end);
      size_t file_size = file.tellg();
      size_t num_entries = file_size / entry_size;
      file.seekg(0, std::ios::beg);

      data_set.reserve(num_entries);
      data_set._dimension = dimension;
      data_set._num = num_entries;

      for (size_t i = 0; i < num_entries; ++i) {
          std::vector<float> vec(dimension);

          file.read(reinterpret_cast<char*>(vec.data()), dimension * sizeof(float));

          data_set._vecs.push_back(vec);
      }
    }

    void ReadPureQuerySet(const std::string &file_name, PureQuerySet &query_set, u_int32_t dimension) {
      std::ifstream file(file_name);
      if (!file.is_open()) {
        throw std::invalid_argument("Cannot open file: " + file_name);
      }
      // Calculate the size of each query entry (one int + 768 floats)
      size_t entry_size = dimension * sizeof(float);

      // Reserve space for the dataset based on the file size
      file.seekg(0, std::ios::end);
      size_t file_size = file.tellg();
      size_t num_entries = file_size / entry_size;
      file.seekg(0, std::ios::beg);

      query_set.reserve(num_entries);
      query_set._dimension = dimension;
      query_set._num = num_entries;

      for (size_t i = 0; i < num_entries; ++i) {
          std::vector<float> vec(dimension);

          file.read(reinterpret_cast<char*>(vec.data()), dimension * sizeof(float));
          query_set._queries.push_back(vec);
      }
    }

    void SaveQueryLabel(const std::string &file_name, const QuerySet &query_set) {
      std::ofstream file(file_name);
      if (!file.is_open()) {
        throw std::invalid_argument("Cannot open file: " + file_name);
      }
      const int num_queries = query_set._num;
      for (size_t i = 0; i < num_queries; ++i) {
        file.write(reinterpret_cast<const char*>(&query_set._queries[i]._label), sizeof(u_int32_t));
      }
    }


    void SaveVectorResult(const std::string &file_name, const std::vector<std::vector<u_int32_t>> &results, const int top_k) {
      std::ofstream file(file_name);
      if (!file.is_open()) {
        throw std::invalid_argument("Cannot open file: " + file_name);
      }
      const int num_queries = results.size();
      for (size_t i = 0; i < num_queries; ++i) {
        file.write(reinterpret_cast<const char*>(&top_k), sizeof(u_int32_t));
        file.write(reinterpret_cast<const char*>(results[i].data()), top_k * sizeof(u_int32_t));
      }
    }

    void SaveLabelResult(const std::string &file_name, const std::vector<std::vector<u_int32_t>> &results, const int top_k) {
      std::ofstream file(file_name);
      if (!file.is_open()) {
        throw std::invalid_argument("Cannot open file: " + file_name);
      }
      const int num_queries = results.size(); 
      for (size_t i = 0; i < num_queries; ++i) {
        file.write(reinterpret_cast<const char*>(&top_k), sizeof(u_int32_t));
        file.write(reinterpret_cast<const char*>(results[i].data()), top_k * sizeof(u_int32_t));
      }
    }
  };

}; 


