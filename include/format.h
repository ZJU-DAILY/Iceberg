#pragma once

#include "common_includes.h"

namespace benchmark {
  class DataSet {    
    public:
    void reserve(size_t size) {
        _vecs.reserve(size);
    }

    void resize(size_t size) {
        _vecs.resize(size);
    }

    size_t size() {
        return _vecs.size();
    }
    
    u_int32_t _dimension;
    u_int32_t _num;
    std::vector<u_int32_t> _label;
    std::vector<std::vector<float>> _vecs;
  };
  
  struct Query{
    u_int32_t _label;
    std::vector<float> _vec;
  };

  class QuerySet {
    public:
    void reserve(size_t size) {
        _queries.reserve(size);
    }

    void resize(size_t size) {
        _queries.resize(size);
    }

    size_t size() {
        return _queries.size();
    }
    u_int32_t _dimension;
    u_int32_t _num;
    std::vector<Query> _queries;
  };
  class PureDataSet {
    public:
    void reserve(size_t size) {
        _vecs.reserve(size);
    }

    void resize(size_t size) {
        _vecs.resize(size);
    }

    size_t size() {
        return _vecs.size();
    }
    u_int32_t _dimension;
    u_int32_t _num;
    std::vector<std::vector<float>> _vecs;
  };
  
  class PureQuerySet {
    public:
    void reserve(size_t size) {
        _queries.reserve(size);
    }

    void resize(size_t size) {
        _queries.resize(size);
    }

    size_t size() {
        return _queries.size();
    }
    u_int32_t _dimension;
    u_int32_t _num;
    std::vector<std::vector<float>> _queries;
  };
  class Parameters {
    public:
    template <typename ParamType>
    inline void Set(const std::string &name, const ParamType &value) {
      std::stringstream sstream;
      sstream << value;
      params[name] = sstream.str();
    }

    inline std::string GetRaw(const std::string &name) const {
      auto item = params.find(name);
      if (item == params.end()) {
        throw std::invalid_argument("Invalid parameter name.");
      } else {
        return item->second;
      }
    }

    template <typename ParamType>
    inline ParamType Get(const std::string &name) const {
      auto item = params.find(name);
      if (item == params.end()) {
        throw std::invalid_argument("Invalid parameter name.");
      } else {
        return ConvertStrToValue<ParamType>(item->second);
      }
    }

    template <typename ParamType>
    inline ParamType Get(const std::string &name,
                        const ParamType &default_value) {
      try {
        return Get<ParamType>(name);
      } catch (std::invalid_argument e) {
        return default_value;
      }
    }

    private:
    std::unordered_map<std::string, std::string> params;

    template <typename ParamType>
    inline ParamType ConvertStrToValue(const std::string &str) const {
      std::stringstream sstream(str);
      ParamType value;
      if (!(sstream >> value) || !sstream.eof()) {
        std::stringstream err;
        err << "Failed to convert value '" << str
            << "' to type: " << typeid(value).name();
        throw std::runtime_error(err.str());
      }
      return value;
    }
  };
};
