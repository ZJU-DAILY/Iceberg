#ifndef EFANNA2E_INDEX_NSG_H
#define EFANNA2E_INDEX_NSG_H

#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"
#include <cassert>
#include <unordered_map>
#include <string>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
#include <stack>
// #include <omp.h>

namespace efanna2e {

class IndexNSG : public Index {
 public:
  IndexNSG(const size_t dimension, const size_t n, Metric m,
    Index *initializer)
    : Index(dimension, n, m), initializer_{initializer} {}


  ~IndexNSG() {

  }

  void Save(const char *filename) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    assert(final_graph_.size() == nd_);
  
    out.write((char *)&width, sizeof(unsigned));
    out.write((char *)&ep_, sizeof(unsigned));
    for (unsigned i = 0; i < nd_; i++) {
      unsigned GK = (unsigned)final_graph_[i].size();
      out.write((char *)&GK, sizeof(unsigned));
      out.write((char *)final_graph_[i].data(), GK * sizeof(unsigned));
    }
    out.close();
  }
  void Load(const char *filename) {
    std::ifstream in(filename, std::ios::binary);
    in.read((char *)&width, sizeof(unsigned));
    in.read((char *)&ep_, sizeof(unsigned));
    // width=100;
    long long avg = 0;
    auto max = 0;
    auto min = 0;
    unsigned cc = 0;
    while (!in.eof()) {
      unsigned k;
      in.read((char *)&k, sizeof(unsigned));
      if (in.eof()) break;
      cc += k;
      std::vector<unsigned> tmp(k);
      in.read((char *)tmp.data(), k * sizeof(unsigned));
      final_graph_.push_back(tmp);
      if (tmp.size() > max) {
        max = tmp.size();
      }
      if (tmp.size() < min || min == 0) {
        min = tmp.size();
      }
      avg += tmp.size();
    }
    cc /= nd_;
    avg /= nd_;
    std::cout << "max length: " << max << std::endl;
    std::cout << "min length: " << min << std::endl;
    std::cout << "avg length: " << avg << std::endl;
  }


  void Build(size_t n, const float *data, const Parameters &parameters) {
    std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
    unsigned range = parameters.Get<unsigned>("R");
    Load_nn_graph(nn_graph_path.c_str());
    data_ = data;
    init_graph(parameters);
    SimpleNeighbor *cut_graph_ = new SimpleNeighbor[1ll * nd_ * (size_t)range];
    Link(parameters, cut_graph_);
    final_graph_.resize(nd_);
  
    for (size_t i = 0; i < nd_; i++) {
      SimpleNeighbor *pool = cut_graph_ + 1ll * i * (size_t)range;
      unsigned pool_size = 0;
      for (unsigned j = 0; j < range; j++) {
        if (pool[j].distance == -1) break;
        pool_size = j;
      }
      pool_size++;
      final_graph_[i].resize(pool_size);
      for (unsigned j = 0; j < pool_size; j++) {
        final_graph_[i][j] = pool[j].id;
      }
    }
  
    // tree_grow(parameters);
  
    unsigned max = 0, min = 1e6, avg = 0;
    for (size_t i = 0; i < nd_; i++) {
      auto size = final_graph_[i].size();
      max = max < size ? size : max;
      min = min > size ? size : min;
      avg += size;
    }
    avg /= 1.0 * nd_;
    printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max, min, avg);
  
    has_built = true;
  }

  void Search(const float *query, const float *x, size_t K,
    const Parameters &parameters, unsigned *indices) {
          int _NDC=0;
          int _NHOPS=0;
          auto start = std::chrono::high_resolution_clock::now();
          const unsigned L = parameters.Get<unsigned>("L_search");
          data_ = x;
          std::vector<Neighbor> retset(L + 1);
          std::vector<unsigned> init_ids(L);
          boost::dynamic_bitset<> flags{nd_, 0};
          // std::mt19937 rng(rand());
          // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

          unsigned tmp_l = 0;
          for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
          init_ids[tmp_l] = final_graph_[ep_][tmp_l];
          flags[init_ids[tmp_l]] = true;
          }

          while (tmp_l < L) {
          unsigned id = rand() % nd_;
          if (flags[id]) continue;
          flags[id] = true;
          init_ids[tmp_l] = id;
          tmp_l++;
          }

          for (unsigned i = 0; i < init_ids.size(); i++) {
            unsigned id = init_ids[i];
            float dist =
            distance_->compare(data_ + 1ll * dimension_ * id, query, (unsigned)dimension_);
            _NDC++;
            retset[i] = Neighbor(id, dist, true);
          // flags[id] = true;
          }

          std::sort(retset.begin(), retset.begin() + L);
          int k = 0;
          while (k < (int)L) {
            int nk = L;

            if (retset[k].flag) {
            retset[k].flag = false;
            unsigned n = retset[k].id;
            _NHOPS++;
            for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
            unsigned id = final_graph_[n][m];
            if (flags[id]) continue;
            flags[id] = 1;
            float dist =
            distance_->compare(query, data_ + 1ll * dimension_ * id, (unsigned)dimension_);_NDC++;
            if (dist >= retset[L - 1].distance) continue;
            Neighbor nn(id, dist, true);
            int r = InsertIntoPool(retset.data(), L, nn);

            if (r < nk) nk = r;
            }
            }
            if (nk <= k)
            k = nk;
            else
            ++k;
          }
         for (size_t i = 0; i < K; i++) {
           indices[i] = retset[i].id;
         }
        // std::cout << "----------"<<K<<"-NN RESULTS----------- "<<std::endl;
        // for(size_t i=0; i < K; i++){
        // printf(" K N°%lu  => Distance : %f | Node ID : %u | Time  : %f | TOTAL DC : %i | Total hops : %i  \n",
        // i+1,std::sqrt(retset[i].distance),retset[i].id,diff.count(),_NDC,_NHOPS);

        // }
        }
  void SearchWithOptGraph(
      const float *query,
      size_t K,
      const Parameters &parameters,
      unsigned *indices) {
        int _NDC=0;
        int _NHOPS=0;
        auto start = std::chrono::high_resolution_clock::now();
      unsigned L = parameters.Get<unsigned>("L_search");
      DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;
    
      std::vector<Neighbor> retset(L + 1);
      std::vector<unsigned> init_ids(L);
      // std::mt19937 rng(rand());
      // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);
    
      boost::dynamic_bitset<> flags{nd_, 0};
      unsigned tmp_l = 0;
      unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * ep_ + data_len);
      unsigned MaxM_ep = *neighbors;
      neighbors++;
    
      for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
        init_ids[tmp_l] = neighbors[tmp_l];
        flags[init_ids[tmp_l]] = true;
      }
    
      while (tmp_l < L) {
        unsigned id = rand() % nd_;
        if (flags[id]) continue;
        flags[id] = true;
        init_ids[tmp_l] = id;
        tmp_l++;
      }
    
      for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        if (id >= nd_) continue;
        _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
      }
      L = 0;
      for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        if (id >= nd_) continue;
        float *x = (float *)(opt_graph_ + node_size * id);
        float norm_x = *x;
        x++;
        _NDC++;
        float dist = dist_fast->compare(x, query, norm_x, (unsigned)dimension_);
        retset[i] = Neighbor(id, dist, true);
        flags[id] = true;
        L++;
      }
      // std::cout<<L<<std::endl;
    
      std::sort(retset.begin(), retset.begin() + L);
      int k = 0;
      while (k < (int)L) {
        int nk = L;
    
        if (retset[k].flag) {
          retset[k].flag = false;
          unsigned n = retset[k].id;
    
          _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
          unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * n + data_len);
          unsigned MaxM = *neighbors;
          neighbors++;
          for (unsigned m = 0; m < MaxM; ++m)
            _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
          _NHOPS++;
          for (unsigned m = 0; m < MaxM; ++m) {
            unsigned id = neighbors[m];
            if (flags[id]) continue;
            flags[id] = 1;
            float *data = (float *)(opt_graph_ + node_size * id);
            float norm = *data;
            data++;
            _NDC++;
            float dist = dist_fast->compare(query, data, norm, (unsigned)dimension_);
            if (dist >= retset[L - 1].distance) continue;
    //        std::cout << dist << std::endl;
            Neighbor nn(id, dist, true);
            int r = InsertIntoPool(retset.data(), L, nn);
    
            // if(L+1 < retset.size()) ++L;
            if (r < nk) nk = r;
          }
        }
        if (nk <= k)
          k = nk;
        else
          ++k;
      }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end-start;
        std::cout << "----------"<<K<<"-NN RESULTS----------- "<<std::endl;
        for(size_t i=0; i < K; i++){
            printf(" K N°%lu  => Distance : %f | Node ID : %u | Time  : %f | TOTAL DC : %i | Total hops : %i  \n",
                   i+1,sqrt(retset[i].distance),retset[i].id,diff.count(),_NDC,_NHOPS);
    
        }
      }
  void OptimizeGraph(float* data) {
    data_ = data;
    data_len = (dimension_ + 1) * sizeof(float);
    neighbor_len = (width + 1) * sizeof(unsigned);
    node_size = data_len + neighbor_len;
    opt_graph_ = (char *)malloc(node_size * nd_);
    DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;
    for (unsigned i = 0; i < nd_; i++) {
      char *cur_node_offset = opt_graph_ + i * node_size;
      float cur_norm = dist_fast->norm(data_ + i * dimension_, dimension_);
      std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
      std::memcpy(cur_node_offset + sizeof(float), data_ + i * dimension_,
                  data_len - sizeof(float));
  
      cur_node_offset += data_len;
      unsigned k = final_graph_[i].size();
      std::memcpy(cur_node_offset, &k, sizeof(unsigned));
      std::memcpy(cur_node_offset + sizeof(unsigned), final_graph_[i].data(),
                  k * sizeof(unsigned));
      std::vector<unsigned>().swap(final_graph_[i]);
    }
    CompactGraph().swap(final_graph_);
  }

  protected:
    typedef std::vector<std::vector<unsigned > > CompactGraph;
    typedef std::vector<SimpleNeighbors > LockGraph;
    typedef std::vector<nhood> KNNGraph;

    CompactGraph final_graph_;

    Index *initializer_;
    void init_graph(const Parameters &parameters) {
        float *center = new float[dimension_];
        for (unsigned j = 0; j < dimension_; j++) center[j] = 0;
        for (unsigned i = 0; i < nd_; i++) {
          for (unsigned j = 0; j < dimension_; j++) {
            center[j] += data_[i * dimension_ + j];
          }
        }
        for (unsigned j = 0; j < dimension_; j++) {
          center[j] /= nd_;
        }
        std::vector<Neighbor> tmp, pool;
        ep_ = rand() % nd_;  // random initialize navigating point
        get_neighbors(center, parameters, tmp, pool);
        ep_ = tmp[0].id;
    }

    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset) {
            unsigned L = parameter.Get<unsigned>("L");

            retset.resize(L + 1);
            std::vector<unsigned> init_ids(L);
            // initializer_->Search(query, nullptr, L, parameter, init_ids.data());
          
            boost::dynamic_bitset<> flags{nd_, 0};
            L = 0;
            for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
              init_ids[i] = final_graph_[ep_][i];
              flags[init_ids[i]] = true;
              L++;
            }
            while (L < init_ids.size()) {
              unsigned id = rand() % nd_;
              if (flags[id]) continue;
              init_ids[L] = id;
              L++;
              flags[id] = true;
            }
          
            L = 0;
            for (unsigned i = 0; i < init_ids.size(); i++) {
              unsigned id = init_ids[i];
              if (id >= nd_) continue;
              // std::cout<<id<<std::endl;
              float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                              (unsigned)dimension_);
              retset[i] = Neighbor(id, dist, true);
              // flags[id] = 1;
              L++;
            }
          
            std::sort(retset.begin(), retset.begin() + L);
            int k = 0;
            while (k < (int)L) {
              int nk = L;
          
              if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;
          
                for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                  unsigned id = final_graph_[n][m];
                  if (flags[id]) continue;
                  flags[id] = 1;
          
                  float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                                  (unsigned)dimension_);
                  Neighbor nn(id, dist, true);
                  fullset.push_back(nn);
                  if (dist >= retset[L - 1].distance) continue;
                  int r = InsertIntoPool(retset.data(), L, nn);
          
                  if (L + 1 < retset.size()) ++L;
                  if (r < nk) nk = r;
                }
              }
              if (nk <= k)
                k = nk;
              else
                ++k;
            }
        }

    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        boost::dynamic_bitset<>& flags,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset) {
            unsigned L = parameter.Get<unsigned>("L");

            retset.resize(L + 1);
            std::vector<unsigned> init_ids(L);
            // initializer_->Search(query, nullptr, L, parameter, init_ids.data());
          
            L = 0;
            // add ep nout to Seeds
            for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
              init_ids[i] = final_graph_[ep_][i];
              flags[init_ids[i]] = true;
              L++;
            }
            // fill the L seeds with random nodes
            while (L < init_ids.size()) {
              unsigned id = rand() % nd_;
              if (flags[id]) continue;
              init_ids[L] = id;
              L++;
              flags[id] = true;
            }
          
            // calculate the distance with L seeds
            L = 0;
            for (unsigned i = 0; i < init_ids.size(); i++) {
              unsigned id = init_ids[i];
              if (id >= nd_) continue;
              // std::cout<<id<<std::endl;
              float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                              (unsigned)dimension_);
              retset[i] = Neighbor(id, dist, true);
              fullset.push_back(retset[i]);
              // flags[id] = 1;
              L++;
            }
          
            std::sort(retset.begin(), retset.begin() + L);
            int k = 0;
            while (k < (int)L) {
              int nk = L;
          
              if (retset[k].flag) {
                retset[k].flag = false;
                unsigned n = retset[k].id;
          
                for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                  unsigned id = final_graph_[n][m];
                  if (flags[id]) continue;
                  flags[id] = 1;
          
                  float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                                  (unsigned)dimension_);
                  Neighbor nn(id, dist, true);
                  fullset.push_back(nn);
                  if (dist >= retset[L - 1].distance) continue;
                  int r = InsertIntoPool(retset.data(), L, nn);
          
                  if (L + 1 < retset.size()) ++L;
                  if (r < nk) nk = r;
                }
              }
              if (nk <= k)
                k = nk;
              else
                ++k;
            }
        }
    //void add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph& cut_graph_);
    void InterInsert(unsigned n, unsigned range, std::vector<std::mutex>& locks, SimpleNeighbor* cut_graph_) {
        SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
        //for every des in nout(n)
        for (size_t i = 0; i < range; i++) {
          if (src_pool[i].distance == -1) break;
      
          //dist between n and des
          SimpleNeighbor sn(n, src_pool[i].distance);
      
          size_t des = src_pool[i].id;
          // get des nout
          SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;
      
          std::vector<SimpleNeighbor> temp_pool;
          int dup = 0;
          {
            LockGuard guard(locks[des]);
            //check nout(des)
            for (size_t j = 0; j < range; j++) {
              if (des_pool[j].distance == -1) break;
              //if n is in nout(des), mode to next nout(n)
              if (n == des_pool[j].id) {
                dup = 1;
                break;
              }
              //if n is not in nout(des), take nout(des) in temp_pool
              temp_pool.push_back(des_pool[j]);
            }
          }
          if (dup) continue;
      // if the connection between n and des is not biderectionnal
      
          // add n to nout(des)
          temp_pool.push_back(sn);
          //if the temp_pool exceed outdegree, apply
          if (temp_pool.size() > range) {
            std::vector<SimpleNeighbor> result;
            unsigned start = 0;
            std::sort(temp_pool.begin(), temp_pool.end());
            result.push_back(temp_pool[start]);
            while (result.size() < range && (++start) < temp_pool.size()) {
              auto &p = temp_pool[start];
              bool occlude = false;
              for (unsigned t = 0; t < result.size(); t++) {
                if (p.id == result[t].id) {
                  occlude = true;
                  break;
                }
                float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                               data_ + dimension_ * (size_t)p.id,
                                               (unsigned)dimension_);
                if (djk < p.distance /* dik */) {
                  occlude = true;
                  break;
                }
              }
              if (!occlude) result.push_back(p);
            }
            {
              LockGuard guard(locks[des]);
              for (unsigned t = 0; t < result.size(); t++) {
                des_pool[t] = result[t];
              }
            }
          }
          else {
            LockGuard guard(locks[des]);
            for (unsigned t = 0; t < range; t++) {
              if (des_pool[t].distance == -1) {
                des_pool[t] = sn;
                if (t + 1 < range) des_pool[t + 1].distance = -1;
                break;
              }
            }
          }
        }        
    }
    void sync_prune(unsigned q, std::vector<Neighbor>& pool, const Parameters &parameter, boost::dynamic_bitset<>& flags, SimpleNeighbor* cut_graph_) {
        unsigned range = parameter.Get<unsigned>("R");
        unsigned maxc = parameter.Get<unsigned>("C");
        width = range;
        unsigned start = 0;
      
        for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
          unsigned id = final_graph_[q][nn];
          if (flags[id]) continue;
          float dist =
              distance_->compare(data_ + dimension_ * (size_t)q,
                                 data_ + dimension_ * (size_t)id, (unsigned)dimension_);
          pool.push_back(Neighbor(id, dist, true));
        }
        std::sort(pool.begin(), pool.end());
      
      
        std::vector<Neighbor> result;
        if (pool[start].id == q) start++;
        result.push_back(pool[start]);
      
        // here where algorithm start
        // while R.size() < M(R) and !|E|.empty // and  we limit start too such that we dont go too deep in visitedd list
        while (result.size() < range && (++start) < pool.size() && start < maxc) {
          auto &p = pool[start];
          bool occlude = false;
          for (unsigned t = 0; t < result.size(); t++) {
            if (p.id == result[t].id) {
              occlude = true;
              break;
            }
            float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                           data_ + dimension_ * (size_t)p.id,
                                           (unsigned)dimension_);
      
            if (djk < p.distance /* dik */) {
              occlude = true;
              break;
            }
          }
          if (!occlude) result.push_back(p);
        }
      
        SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
        for (size_t t = 0; t < result.size(); t++) {
          des_pool[t].id = result[t].id;
          des_pool[t].distance = result[t].distance;
        }
        if (result.size() < range) {
          des_pool[result.size()].distance = -1;
        }
    }
    void Link(const Parameters &parameters, SimpleNeighbor* cut_graph_) {
        unsigned range = parameters.Get<unsigned>("R");
        std::vector<std::mutex> locks(nd_);
      std::cout << "Linking graph with R = " << range << std::endl;
      #pragma omp parallel for
          for (unsigned n = 0; n < nd_; ++n) {
            std::vector<Neighbor> pool, tmp;
            boost::dynamic_bitset<> flags{nd_, 0};
            pool.clear();
            tmp.clear();
            flags.reset();
            // perform beam search, flags is a bitmap of visited node, tmp is L results, and pool is checked nodes
            get_neighbors(data_ + 1ll * dimension_ * n, parameters, flags, tmp, pool);
            sync_prune(n, pool, parameters, flags, cut_graph_);
          }
        
      std::cout << "Interlinking graph with R = " << range << std::endl;
      #pragma omp parallel for
        for (unsigned n = 0; n < nd_; ++n) {
          InterInsert(n, range, locks, cut_graph_);
        }  
    }

    void Load_nn_graph(const char *filename) {
        std::ifstream in(filename, std::ios::binary);
        unsigned k;
        in.read((char *)&k, sizeof(unsigned));
        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        size_t fsize = (size_t)ss;
        size_t num = (unsigned)(fsize / (k + 1) / 4);
        in.seekg(0, std::ios::beg);
      
        final_graph_.resize(num);
        final_graph_.reserve(num);
        unsigned kk = (k + 3) / 4 * 4;
        for (size_t i = 0; i < num; i++) {
          in.seekg(4, std::ios::cur);
          final_graph_[i].resize(k);
          final_graph_[i].reserve(kk);
          in.read((char *)final_graph_[i].data(), k * sizeof(unsigned));
        }
        in.close();
    }
    
    void tree_grow(const Parameters &parameter) {
        unsigned root = ep_;
        boost::dynamic_bitset<> flags{nd_, 0};
        unsigned unlinked_cnt = 0;
        while (unlinked_cnt < nd_) {
          DFS(flags, root, unlinked_cnt);
          // std::cout << unlinked_cnt << '\n';
          if (unlinked_cnt >= nd_) break;
          findroot(flags, root, parameter);
          // std::cout << "new root"<<":"<<root << '\n';
        }
        for (size_t i = 0; i < nd_; ++i) {
          if (final_graph_[i].size() > width) {
            width = final_graph_[i].size();
          }
        }
    }
    void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
        unsigned tmp = root;
        std::stack<unsigned> s;
        s.push(root);
        if (!flag[root]) cnt++;
        flag[root] = true;
        while (!s.empty()) {
          unsigned next = nd_ + 1;
          for (unsigned i = 0; i < final_graph_[tmp].size(); i++) {
            if (flag[final_graph_[tmp][i]] == false) {
              next = final_graph_[tmp][i];
              break;
            }
          }
          // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
          if (next == (nd_ + 1)) {
            s.pop();
            if (s.empty()) break;
            tmp = s.top();
            continue;
          }
          tmp = next;
          flag[tmp] = true;
          s.push(tmp);
          cnt++;
        }
    }
    void findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter) {
        unsigned id = nd_;
        for (unsigned i = 0; i < nd_; i++) {
          if (flag[i] == false) {
            id = i;
            break;
          }
        }
      
        if (id == nd_) return;  // No Unlinked Node
      
        std::vector<Neighbor> tmp, pool;
        get_neighbors(data_ + dimension_ * id, parameter, tmp, pool);
        std::sort(pool.begin(), pool.end());
      
        unsigned found = 0;
        for (unsigned i = 0; i < pool.size(); i++) {
          if (flag[pool[i].id]) {
            // std::cout << pool[i].id << '\n';
            root = pool[i].id;
            found = 1;
            break;
          }
        }
        if (found == 0) {
          while (true) {
            unsigned rid = rand() % nd_;
            if (flag[rid]) {
              root = rid;
              break;
            }
          }
        }
        final_graph_[root].push_back(id);
    }


  private:
    unsigned width;
    unsigned ep_;
    std::vector<std::mutex> locks;
    char* opt_graph_;
    size_t node_size;
    size_t data_len;
    size_t neighbor_len;
    KNNGraph nnd_graph;
};
}

#endif //EFANNA2E_INDEX_NSG_H
