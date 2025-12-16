#ifndef EFANNA2E_INDEX_H
#define EFANNA2E_INDEX_H

#include <cstddef>
#include <string>
#include <vector>
#include <fstream>
#include "distance.h"
#include "parameters.h"

namespace efanna2e {

class Index {
 public:
  Index(const size_t dimension, const size_t n, Metric metric) : dimension_ (dimension), nd_(n), has_built(false) {
    switch (metric) {
      case L2:distance_ = new DistanceL2();
        break;
      default:distance_ = new DistanceL2();
        break;
    }
  }


  ~Index() {

  }

  void Build(size_t n, const float *data, const Parameters &parameters);

  void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices);

  void Save(const char *filename);

  void Load(const char *filename);

  inline bool HasBuilt() const { return has_built; }

  inline size_t GetDimension() const { return dimension_; };

  inline size_t GetSizeOfDataset() const { return nd_; }

  inline const float *GetDataset() const { return data_; }
 protected:
  const size_t dimension_;
  const float *data_;
  size_t nd_;
  bool has_built;
  Distance* distance_;
};

}

#endif //EFANNA2E_INDEX_H
