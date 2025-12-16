#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <queue>
#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <unordered_map>
#include <string>
#include <sstream>
#include <stack>
#include <mutex>
#include <iostream>
#include <fstream>
#include <set>

#include "distance.h"
#include "parameters.h"
#include "neigbor.h"


namespace MAG {
#define _CONTROL_NUM 100
  typedef std::vector<std::vector<unsigned > > CompactGraph;
  class IndexMAG {
    public:
      const size_t dimension_;
      const float *data_;
      size_t nd_;
      bool has_built;
      Distance *distance_;
      DistanceInnerProduct *distance_ip_;
      unsigned width;
      unsigned ep_;
      std::vector<std::mutex> locks;
      char* opt_graph_ = nullptr;
      size_t node_size;
      size_t data_len;
      size_t neighbor_len;
      CompactGraph final_graph_;
      CompactGraph ip_graph_;
      std::vector<bool> is_out_dominantor_;
      std::vector<bool> is_self_dominantor_;
      std::vector<std::pair<float, unsigned>> entries;
      

      IndexMAG(const size_t dimension, const size_t n): dimension_(dimension), nd_(n), has_built(false) {
        data_ = nullptr;
        distance_ = new DistanceL2();
        distance_ip_ = new DistanceInnerProduct();
        is_out_dominantor_.resize(nd_);
        is_self_dominantor_.resize(nd_);
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
        unsigned cc = 0;
        while (!in.eof()) {
          unsigned k;
          in.read((char *)&k, sizeof(unsigned));
          if (in.eof()) break;
          cc += k;
          std::vector<unsigned> tmp(k);
          in.read((char *)tmp.data(), k * sizeof(unsigned));
          final_graph_.push_back(tmp);
        }
        cc /= nd_;

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

      void get_nn_neighbors(const float *query, const Parameters &parameter,
                                  boost::dynamic_bitset<> &flags,
                                  std::vector<Neighbor> &retset,
                                  std::vector<Neighbor> &fullset) {
        unsigned L = parameter.Get<unsigned>("L");

        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

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

      void get_ip_neighbors(const float *query, const Parameters &parameter,
                                  boost::dynamic_bitset<> &flags,
                                  std::vector<IpNeighbor> &retset,
                                  std::vector<IpNeighbor> &fullset) {
        unsigned L = parameter.Get<unsigned>("L");

        retset.resize(L + 1);
        std::vector<unsigned> init_ids(L);
        // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

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
          float dist = distance_ip_->compare(data_ + dimension_ * (size_t)id, query,
                                          (unsigned)dimension_);
          retset[i] = IpNeighbor(id, dist, true);
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

              float dist = distance_ip_->compare(query, data_ + dimension_ * (size_t)id,
                                              (unsigned)dimension_);
              IpNeighbor nn(id, dist, true);
              fullset.push_back(nn);
              if (dist <= retset[L - 1].distance) continue;
              int r = InsertIntoIpPool(retset.data(), L, nn);

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
      void get_neighbors(const float *query, const Parameters &parameter,
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
        delete center;
      }

      void sync_prune(unsigned q, std::vector<Neighbor> &pool,
                                const Parameters &parameter,
                                boost::dynamic_bitset<> &flags,
                                SimpleNeighbor *cut_graph_) {
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

      void InterInsert(unsigned n, unsigned range,
                                std::vector<std::mutex> &locks,
                                SimpleNeighbor *cut_graph_) {
        SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
        for (size_t i = 0; i < range; i++) {
          if (src_pool[i].distance == -1) break;

          SimpleNeighbor sn(n, src_pool[i].distance);
          size_t des = src_pool[i].id;
          SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

          std::vector<SimpleNeighbor> temp_pool;
          int dup = 0;
          {
            LockGuard guard(locks[des]);
            for (size_t j = 0; j < range; j++) {
              if (des_pool[j].distance == -1) break;
              if (n == des_pool[j].id) {
                dup = 1;
                break;
              }
              temp_pool.push_back(des_pool[j]);
            }
          }
          if (dup) continue;

          temp_pool.push_back(sn);
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
          } else {
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

      void Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) {
        unsigned range = parameters.Get<unsigned>("R");
        std::vector<std::mutex> locks(nd_);

        #pragma omp parallel
          {
            // unsigned cnt = 0;
            std::vector<Neighbor> pool, tmp;
            boost::dynamic_bitset<> flags{nd_, 0};
        #pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < nd_; ++n) {
              pool.clear();
              tmp.clear();
              flags.reset();
              get_nn_neighbors(data_ + dimension_ * n, parameters, flags, tmp, pool);
              sync_prune(n, pool, parameters, flags, cut_graph_);
            }
          }
        std::cout << "sync prune done!" << std::endl;
        #pragma omp parallel for schedule(dynamic, 100)
          for (unsigned n = 0; n < nd_; ++n) {
            InterInsert(n, range, locks, cut_graph_);
          }
        std::cout << "inter insert done!" << std::endl;
        }
        std::vector<unsigned> pruneEdge(unsigned cur_point, const Parameters &parameters, std::vector<IpNeighbor> &pool, const unsigned threshold) {
          unsigned start = 0;
          unsigned R_IP = parameters.Get<unsigned>("R_IP");
          boost::dynamic_bitset<> flags(nd_, 0);
          std::sort(pool.begin(), pool.end());
          std::vector<IpNeighbor> result;
          std::unordered_map<unsigned,float> self_dist_map;
          unsigned real_m = 0;
          auto cur_ip = distance_ip_->compare(data_ + dimension_ * cur_point,
                                            data_ + dimension_ * cur_point,
                                            (unsigned)dimension_);
          
          if (cur_ip >= pool[start].distance) {
            is_self_dominantor_[cur_point] = true;
          }

          while (start < pool.size() && real_m < threshold) {
            if (pool[start].id == cur_point) {
              start++;
              continue;
            }
            is_out_dominantor_[pool[start].id] = true; 
            result.push_back(pool[start]);
            real_m++;

            auto ip_self = distance_ip_->compare(data_ + dimension_ * pool[start].id,
                                              data_ + dimension_ * pool[start].id,
                                              (unsigned)dimension_);
            self_dist_map[pool[start].id] = ip_self;  
            start++;      
          }

          while (real_m < R_IP && (++start) < pool.size()) {
            if(pool[start].id == cur_point) {
              continue;
            }
            bool occlude = false;
            auto &p = pool[start];

            auto ip_self = distance_ip_->compare(data_ + dimension_ * p.id,
                                              data_ + dimension_ * p.id,
                                              (unsigned)dimension_);
            self_dist_map[p.id] = ip_self;
            for (auto i = 0; i < result.size(); i++) {
              auto nid = result[i].id;
              if (flags[nid]) {
                continue;
              }
              
              auto ip = distance_ip_->compare(data_ + dimension_ * p.id,
                                  data_ + dimension_ * nid,
                                  (unsigned)dimension_);

              if (ip_self < ip) {
                occlude = true;
                break;
              }
              if (self_dist_map[nid] < ip && real_m > threshold) {
                flags[nid] = true;
                real_m--;
              }
            }
            if (!occlude) {
              result.push_back(p);
              real_m++;
            }
          }
          std::vector<unsigned> prune_result;
          for (auto i = 0; i < result.size(); i++) {
            if (prune_result.size() >= R_IP) {
              break;
            }
            if (flags[result[i].id]) {
              continue;
            }
            prune_result.push_back(result[i].id);
          }
          return prune_result;
        }

        void Build(size_t n, const float *data, const Parameters &parameters) {
          std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
          unsigned range = parameters.Get<unsigned>("R");
          unsigned threshold = parameters.Get<unsigned>("threshold");
          Load_nn_graph(nn_graph_path.c_str());
          data_ = data;
          init_graph(parameters);
          std::cout << "load nn graph!" << std::endl;
          SimpleNeighbor *cut_graph_ = new SimpleNeighbor[nd_ * (size_t)range];
          Link(parameters, cut_graph_);
          std::cout << "Link done!" << std::endl;
          final_graph_.resize(nd_);

          for (size_t i = 0; i < nd_; i++) {
            SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
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

          ip_graph_.resize(nd_);
          
          #pragma omp parallel
          {
            std::vector<IpNeighbor> pool, tmp;
            boost::dynamic_bitset<> flags{nd_, 0};
        #pragma omp for schedule(dynamic, 100)
            for (unsigned n = 0; n < nd_; ++n) {
              pool.clear();
              tmp.clear();
              flags.reset();
              get_ip_neighbors(data_ + dimension_ * n, parameters, flags, tmp, pool);
              ip_graph_[n] = pruneEdge(n, parameters, pool, threshold);
            }
          }
          CompactGraph mix_graph;
          mix_graph.resize(nd_);
          for (size_t i = 0; i < nd_; i++) {
            std::set<unsigned> dup;  
            for (auto neighbor: final_graph_[i]) {
              // mix_graph[i].push_back(neighbor);
              dup.insert(neighbor);
            }
            for (auto neighbor: ip_graph_[i]){
              dup.insert(neighbor);
              if (dup.size() >= range) {
                break;
              }
            }

            final_graph_[i].clear();
            for(auto neighbor : dup) {
              final_graph_[i].emplace_back(neighbor);
            }
          }

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
          delete cut_graph_;
        }      

        void entry_point_candidate(float * data_) {
          entries.clear();

          for (auto i = 0; i < nd_; i ++) {
            float norm = distance_ip_->compare(data_ + dimension_ * (size_t)i,
                                              data_ + dimension_ * (size_t)i,
                                              (unsigned)dimension_);
            entries.emplace_back(norm, i);
          }
          std::sort(entries.begin(), entries.end());
        }

        int Search_NN_IP(const float *query, const float *x, size_t K,
                              const Parameters &parameters, unsigned *indices) {
          
          const unsigned L_NN = 5;
          std::vector<Neighbor> retset_nn(L_NN + 1);
          std::vector<unsigned> init_nn_ids(L_NN);
          boost::dynamic_bitset<> flags{nd_, 0};
          auto dis_cal = 0;
          data_ = x;
          for(unsigned i=0; i<final_graph_[ep_].size() && i < L_NN; i++){
            init_nn_ids[i] = final_graph_[ep_][i];
          }

          for (unsigned i = 0; i < L_NN; i++) {
            unsigned id = init_nn_ids[i];
            float nn = distance_->compare(data_ + dimension_ * id, query,
                                            dimension_);
            retset_nn[i] = Neighbor(id, nn, true);
            flags[id] = true;
          }

          std::sort(retset_nn.begin(), retset_nn.begin() + L_NN);

          
          int k = 0;
          while (k < (int)L_NN) {
            int nk = L_NN;
            if (retset_nn[k].flag) {
              retset_nn[k].flag = false;
              unsigned n = retset_nn[k].id;
              int search_pos = final_graph_[n].size();
              for (unsigned m = 0; m < search_pos; ++m) {
                unsigned id = final_graph_[n][m];
                if (flags[id]) continue;
                flags[id] = 1;        
                float dist = distance_->compare(query, data_ + dimension_ * id,
                                                dimension_); 
                dis_cal += 1;
                if (dist >= retset_nn[L_NN - 1].distance) continue;
                Neighbor nn(id, dist, true);
                int r = InsertIntoPool(retset_nn.data(), L_NN, nn);
                if (r < nk) {
                  nk = r;
                }
              }
            }
            if (nk <= k)
              k = nk;
            else
              ++k;
          }

          // =================== swith step ==================
          const unsigned L = parameters.Get<unsigned>("L_search");
          flags.reset();

          std::vector<IpNeighbor> retset(L + 1);
          std::vector<unsigned> init_ids(L);

          std::mt19937 rng(rand());
          GenRandom(rng, init_ids.data(), L, (unsigned)nd_);

          for(unsigned i=0; i < L_NN; i++) {
            init_ids[i] = retset_nn[i].id;
          }
          for(unsigned i=L_NN; i < L ; i++) {
            if(entries[i].second <= nd_) {
              init_ids[i] = entries[i].second;  
            }
          }

          for (unsigned i = 0; i < L; i++) {
            unsigned id = init_ids[i];
            float ip = distance_ip_->compare(data_ + dimension_ * id, query,
                                            dimension_);
            retset[i] = IpNeighbor(id, ip, true);
            flags[id] = true;
          }

          std::sort(retset.begin(), retset.begin() + L);
          
          k = 0;
          while (k < (int)L) {
            int nk = L;
            if (retset[k].flag) {
              retset[k].flag = false;
              unsigned n = retset[k].id;
              int search_pos = final_graph_[n].size();
              for (unsigned m = 0; m < search_pos; ++m) {
                unsigned id = final_graph_[n][m];
                if (flags[id]) continue;
                flags[id] = 1;        
                float dist = distance_ip_->compare(query, data_ + dimension_ * id,
                                                dimension_); 
                dis_cal += 1;
                if (dist <= retset[L - 1].distance) continue;
                IpNeighbor nn(id, dist, true);
                int r = InsertIntoIpPool(retset.data(), L, nn);
                if (r < nk) {
                  nk = r;
                }
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
          return {dis_cal};
      }
  };



}