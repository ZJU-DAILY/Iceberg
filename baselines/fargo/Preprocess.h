#pragma once
#include "fargo/space_l2.h"
#include "fargo/space_ip.h"
#include "format.h"
#include <cmath>
#include <assert.h>
#include <string>
#include <vector>

#define CANDIDATES 100
#define E 2.718281746
#define PI 3.1415926
#define MAXSIZE 40960

namespace fargo {
	struct Data {
		// Dimension of data
		int dim;
		// Number of data
		int N;
		// Data matrix
		float** val;
	};

	struct HashParam {
		// the value of a in S hash functions
		int S;
		int dim;
		float** rndAs1;
		//S*dim
		// the value of a in S hash functions
		float** rndAs2;
		//S*1
		void saveIndex (std::ofstream& output) {
			for(int i = 0;i < S;i++) {
				writeBinaryPOD(output, rndAs2[i][0]);
				for(int j = 0;j < dim;j++) {
					writeBinaryPOD(output, rndAs1[i][j]);
				}
			}
		}
		void loadIndex (std::ifstream& input) {
			rndAs1 = new float* [S];
			rndAs2 = new float* [S];
			for(int i = 0;i < S;i++) {
				rndAs2[i] = new float [1];
				rndAs1[i] = new float [dim];
				readBinaryPOD(input, rndAs2[i][0]);
				for(int j = 0;j < dim;j++) {
					readBinaryPOD(input, rndAs1[i][j]);
				}
			}
		}
	};
	template <class T>
	void clear_2d_array(T** array, int n) {
		for (int i = 0; i < n; ++i) {
			delete[] array[i];
		}
		delete[] array;
	}

	class Preprocess {
	public:
		Data data;
		Data query;
		DISTFUNC<float> fstdistfunc_;
		void *dist_func_param_{nullptr};
		float* SquareLen;
		float MaxLen;
		int top_k;
	public:
		void get_val(Data& _data, std::vector<benchmark::Query>& _vecs) {
			_data.val = new float*[_data.N];
			for(size_t i = 0;i < _data.N;i++) {
				_data.val[i] = _vecs[i]._vec.data();
			}
		}
		void get_val(Data& _data, std::vector<std::vector<float>>& _vecs) {
			_data.val = new float*[_data.N];
			for(size_t i = 0;i < _data.N;i++) {
				_data.val[i] = _vecs[i].data();
			}
		}
		Preprocess(benchmark::DataSet& dataset_, benchmark::QuerySet& queryset_, fargo::SpaceInterface<float>* s, u_int32_t &top_k_) {
			data.dim = dataset_._dimension;
			data.N = dataset_._num;
			get_val(data, dataset_._vecs);
			query.dim = queryset_._dimension;
			query.N = queryset_._num;
			get_val(query, queryset_._queries);
			top_k = top_k_;
			fstdistfunc_ = s->get_dist_func();
  	  		dist_func_param_ = s->get_dist_func_param();
			cal_SquareLen();
		}

		void cal_SquareLen() {
			SquareLen = new float[data.N];
			for (int i = 0; i < data.N; ++i) {
				SquareLen[i] = 1.0 - fstdistfunc_(data.val[i], data.val[i], dist_func_param_);
			}

			MaxLen = *std::max_element(SquareLen, SquareLen + data.N);
		}
		~Preprocess() {
			delete[] SquareLen;
		}

	};

	struct Dist_id {
		int id;
		float dist;
		bool operator < (const Dist_id& rhs) {
			return dist < rhs.dist;
		}
	};

	class Partition {
	private:
		float ratio;
		void MakeChunks(Preprocess& prep) {
			distpairs.clear();
			std::vector<int> bucket;
			Dist_id pair;
			int N_ = prep.data.N;
			int n;
			for (int j = 0; j < N_; j++) {
				pair.id = j;
				pair.dist = prep.SquareLen[j];
				distpairs.push_back(pair);
				distpairs_size++;
				tot_size += sizeof(float) + sizeof(int);
			}
			std::sort(distpairs.begin(), distpairs.end());

			num_chunk = 0;
			chunks.resize(N_);
			chunks_size = N_;
			tot_size += sizeof(int) * N_;
			int j = 0;
			while (j < N_) {
				float M = distpairs[j].dist / ratio;
				n = 0;
				bucket.clear();
				while (j < N_) {
					if ((distpairs[j].dist > M || n >= MAXSIZE)) {
						break;
					}

					chunks[distpairs[j].id] = num_chunk;
					bucket.push_back(distpairs[j].id);
					tot_size += sizeof(int);
					j++;
					n++;
				}
				nums.push_back(n);
				nums_size++;
				tot_size += sizeof(int);
				MaxLen.push_back(distpairs[(size_t)j - 1].dist);
				MaxLen_size++;
				tot_size += sizeof(float);
				EachParti.push_back(bucket);
				EachParti_size.push_back(bucket.size());
				tot_size += sizeof(int);
				bucket.clear();
				num_chunk++;
			}

			display();
		}
	public:
		int num_chunk;

		int tot_size;
		//N

		int MaxLen_size;
		std::vector<float> MaxLen;
		//The chunk where each point belongs
		int chunks_size;
		std::vector<int> chunks;
		//The data size of each chunks
		int nums_size;
		std::vector<int> nums;
		//The buckets by parti;
		std::vector<int> EachParti_size;
		std::vector<std::vector<int>> EachParti;
		//The size of first dimension of EachParti is MaxLen_size
		int distpairs_size;
		std::vector<Dist_id> distpairs;
		void saveIndex (std::ofstream& output) {
			writeBinaryPOD(output, ratio);
			writeBinaryPOD(output, num_chunk);
			writeBinaryPOD(output, tot_size);

			writeBinaryPOD(output, MaxLen_size);
			for(int i = 0;i < MaxLen_size;i++)
				writeBinaryPOD(output, MaxLen[i]);

			writeBinaryPOD(output, chunks_size);
			for(int i = 0;i < chunks_size;i++)
				writeBinaryPOD(output, chunks[i]);

			writeBinaryPOD(output, nums_size);
			for(int i = 0;i < nums_size;i++)
				writeBinaryPOD(output, nums[i]);

			for(int i = 0;i < MaxLen_size;i++) {
				writeBinaryPOD(output, EachParti_size[i]);
				for(int j = 0;j < EachParti_size[i];j++)
					writeBinaryPOD(output, EachParti[i][j]);
			}

			writeBinaryPOD(output, distpairs_size);
			for(int i = 0;i < distpairs_size;i++) {
				writeBinaryPOD(output, distpairs[i].id);
				writeBinaryPOD(output, distpairs[i].dist);
			}
		}
		void loadIndex (std::ifstream& input) {
			readBinaryPOD(input, ratio);
			readBinaryPOD(input, num_chunk);
			readBinaryPOD(input, tot_size);

			readBinaryPOD(input, MaxLen_size);
			MaxLen.resize(MaxLen_size);
			for(int i = 0;i < MaxLen_size;i++){
				readBinaryPOD(input, MaxLen[i]);
			}

			readBinaryPOD(input, chunks_size);
			chunks.resize(chunks_size);
			for(int i = 0;i < chunks_size;i++)
				readBinaryPOD(input, chunks[i]);

			readBinaryPOD(input, nums_size);
			nums.resize(nums_size);
			for(int i = 0;i < nums_size;i++)
				readBinaryPOD(input, nums[i]);

			EachParti.resize(MaxLen_size);
			EachParti_size.resize(MaxLen_size);
			for(int i = 0;i < MaxLen_size;i++) {
				readBinaryPOD(input, EachParti_size[i]);
				EachParti[i].resize(EachParti_size[i]);
				for(int j = 0;j < EachParti_size[i];j++)
					readBinaryPOD(input, EachParti[i][j]);
			}

			readBinaryPOD(input, distpairs_size);
			distpairs.resize(distpairs_size);
			for(int i = 0;i < distpairs_size;i++) {
				readBinaryPOD(input, distpairs[i].id);
				readBinaryPOD(input, distpairs[i].dist);
			}
		}
		void display() {
			std::vector<int> n_(num_chunk, 0);
			int N_ = std::accumulate(nums.begin(), nums.end(), 0);
			for (int j = 0; j < N_; j++) {
				n_[chunks[j]]++;
			}
			bool f1 = false, f2 = false;
			for (int j = 0; j < num_chunk; j++) {
				if (n_[j] != nums[j]) {
					f1 = true;
					break;
				}
			}

		}
		Partition(){}

		Partition(float c_, Preprocess& prep) {
			MaxLen_size = 0;
			chunks_size = 0;
			nums_size = 0;
			distpairs_size = 0;
			ratio = 0.95;
			float c0_ = 1.5f;
			tot_size = sizeof(int) * 6 + sizeof(float);
			MakeChunks(prep);
		}

		Partition(float c_, float c0_, Preprocess& prep) {
			ratio = (pow(c0_, 4.0f) - 1) / (pow(c0_, 4.0f) - c_);
			MakeChunks(prep);
		}
		~Partition(){}
	};


	class Parameter {//N,dim,S, L, K, M, W;
	public:
		int N;
		int dim;
		// Number of hash functions
		int S;
		//#L Tables; 
		int L;
		// Dimension of the hash table
		int K;
		//
		int MaxSize;
		//
		int KeyLen;

		int M = 1;

		int W = 0;

		float U;
		Parameter(Preprocess& prep, u_int32_t L_, u_int32_t K_) {
			N = prep.data.N;
			dim = prep.data.dim;
			L = L_;
			K = K_;
			S = L * K;
			MaxSize = 3;
			KeyLen = 1;
		}
		Parameter(Preprocess& prep, int L_, int K_, int M_, float U_) {
			N = prep.data.N;
			dim = prep.data.dim;
			M = M_;
			L = L_;
			K = K_;
			S = L * K;
			U = U_;
		}
		Parameter(Preprocess& prep, int L_, int K_, int M_, float U_, float W_) {
			N = prep.data.N;
			dim = prep.data.dim;
			L = L_;
			K = K_;
			S = L * K;
			U = U_;
			W = W_;
		}
		Parameter(Preprocess& prep, float c_, float S0) {

			N = prep.data.N;
			dim = prep.data.dim;
			assert(c_ * S0 < 1);
			double pi = atan(1) * 4;
			double p1 = 1 - acos(S0) / pi;
			double p2 = 1 - acos(c_ * S0) / pi;
			K = (int)floor(log(N) / log(1 / p2)) + 1;
			double rho = log(p1) / log(p2);
			L = (int)floor(pow((double)N, rho)) + 1;
			S = L * K;

			
		}
		inline float normal_pdf0(			// pdf of Guassian(mean, std)
			float x,							// variable
			float u,							// mean
			float sigma) {					// standard error
			float ret = exp(-(x - u) * (x - u) / (2.0f * sigma * sigma));
			ret /= sigma * sqrt(2.0f * PI);
			return ret;
		}

		float new_cdf0(						// cdf of N(0, 1) in range [-x, x]
			float x,							// integral border
			float step) {						// step increment
			float result = 0.0f;
			for (float i = -x; i <= x; i += step) {
				result += step * normal_pdf0(i, 0.0f, 1.0f);
			}
			return result;
		}

		inline float calc_p0(			// calc probability
			float x) {						// x = w / (2.0 * r)
			return new_cdf0(x, 0.001f);		// cdf of [-x, x]
		}
		Parameter(Preprocess& prep, float c_) {
			K = 0;
			L = 0;
			N = prep.data.N;
			dim = prep.data.dim;

			KeyLen = -1;
			MaxSize = -1;
			float w_ = sqrt((8.0f * c_ * c_ * log(c_)) / (c_ * c_ - 1.0f));

			float beta_;
			float delta_;
			float p1_;
			float p2_;
			float para1;
			float para2;
			float para3;
			float eta;
			float alpha_;
			float m, l;

			int n_pts_ = MAXSIZE;
			beta_ = (float)CANDIDATES / n_pts_;
			delta_ = 1.0f / E;

			p1_ = calc_p0(w_ / 2.0f);
			p2_ = calc_p0(w_ / (2.0f * c_));

			para1 = sqrt(log(2.0f / beta_));
			para2 = sqrt(log(1.0f / delta_));
			para3 = 2.0f * (p1_ - p2_) * (p1_ - p2_);
			eta = para1 / para2;

			alpha_ = (eta * p1_ + p2_) / (1.0f + eta);
			m = (para1 + para2) * (para1 + para2) / para3;
			this->S = (int)ceil(m);
			M = 1;
			W = 1;
		}
		bool operator = (const Parameter& rhs) {
			bool flag = 1;
			flag *= (L == rhs.L);
			flag *= (K = rhs.K);
			return flag;
		}

		~Parameter(){}
	};


}