#pragma once
#include "fargo/Preprocess.h"
#include <cmath>
#include <assert.h>
#include <vector>
#include <queue>
#include <cfloat>

namespace fargo {

// #define MAXSIZE 40960
#define pi 3.141592653
#define CANDIDATES 100
#define MINFLOAT -3.40282e+038

	namespace mf_alsh {
		class Hash {
			
		public:
			int N;
			int dim;
			// Number of hash functions
			int S;
			//#L Tables; 
			int L;
			// Dimension of the hash table
			int K;

			long long tot_size;
			//dim*S+num_chunk*L*(1<<K)+L*N+parti.tot_size
			//S=L*K
			//parti.tot_size=N
			//myIndexes(N*L+num_chunk*L*(1<<K)) takes up most of the memory

			float** hashval = nullptr;
			//No storage required

			Partition parti;
			HashParam hashpar;
			int ***myIndexes_size;
			std::vector<int>*** myIndexes;

			float tmin;
			float tstep;
			float smin;
			float sstep;
			int rows;
			int cols;
			//hyperparameter

			float** phi;
			
			void saveIndex(const std::string& file) {
				std::ofstream output(file, std::ios::binary);
				writeBinaryPOD(output, N);
				writeBinaryPOD(output, dim);
				writeBinaryPOD(output, S);
				writeBinaryPOD(output, L);
				writeBinaryPOD(output, K);
				writeBinaryPOD(output, tot_size);
				writeBinaryPOD(output, tmin);
				writeBinaryPOD(output, tstep);
				writeBinaryPOD(output, smin);
				writeBinaryPOD(output, sstep);
				writeBinaryPOD(output, rows);
				writeBinaryPOD(output, cols);
				parti.saveIndex(output);
				hashpar.saveIndex(output);
				for(int i = 0;i < parti.num_chunk;i++)
					for(int j = 0;j < L;j++)
						for(int k = 0;k < (1 << K);k++) {
							writeBinaryPOD(output, myIndexes_size[i][j][k]);
							for(int l = 0;l < myIndexes_size[i][j][k];l++)
								writeBinaryPOD(output, myIndexes[i][j][k][l]);
						}
				for(int i = 0;i < rows;i++) {
					for(int j = 0;j < cols;j++) {
						writeBinaryPOD(output, phi[i][j]);
					}
				}
				output.close();
			}

			void loadIndex(const std::string& file) {
				std::ifstream input(file, std::ios::binary);
				readBinaryPOD(input, N);
				readBinaryPOD(input, dim);
				readBinaryPOD(input, S);
				hashpar.S = S;
				hashpar.dim = dim;
				readBinaryPOD(input, L);
				readBinaryPOD(input, K);
				readBinaryPOD(input, tot_size);
				readBinaryPOD(input, tmin);
				readBinaryPOD(input, tstep);
				readBinaryPOD(input, smin);
				readBinaryPOD(input, sstep);
				readBinaryPOD(input, rows);
				readBinaryPOD(input, cols);
				parti.loadIndex(input);
				hashpar.loadIndex(input);
				myIndexes_size = new int** [parti.num_chunk];
				myIndexes = new std::vector<int>** [parti.num_chunk];
				for(int i = 0;i < parti.num_chunk;i++) {
					myIndexes_size[i] = new int* [L];
					myIndexes[i] = new std::vector<int>* [L];
					for(int j = 0;j < L;j++) {
						myIndexes_size[i][j] = new int [1 << K];
						myIndexes[i][j] = new std::vector<int> [1 << K];
						for(int k = 0;k < (1 << K);k++) {
							readBinaryPOD(input, myIndexes_size[i][j][k]);
							myIndexes[i][j][k].resize(myIndexes_size[i][j][k]);
							for(int l = 0;l < myIndexes_size[i][j][k];l++)
								readBinaryPOD(input, myIndexes[i][j][k][l]);
						}
					}
				}
				phi = new float* [rows];
				for(int i = 0;i < rows;i++) {
					phi[i] = new float [cols];
					for(int j = 0;j < cols;j++) {
						readBinaryPOD(input, phi[i][j]);
					}
				}
				input.close();
			}

			void load_funtable(const std::string& file) {
				std::ifstream is(file.c_str(), std::ios::binary);
				float header1[4] = { 0 };
				int header2[3] = { 0 };
				assert(sizeof header1 == 4 * 4);
				is.read((char*)header1, sizeof(header1));
				assert(sizeof header2 == 3 * 4);
				is.read((char*)header2, sizeof(header2));
				assert(header2[1] != 0);

				rows = header2[1];
				cols = header2[2];

				tot_size += 2 * sizeof(int);

				tmin = header1[0];
				tstep = (header1[1] - header1[0]) / rows;
				smin = header1[2];
				sstep = (header1[3] - header1[2]) / cols;

				tot_size += 4 * sizeof(float);

				float* array_ = new float[(size_t)header2[2] * header2[1]];
				is.read((char*)&array_[0], sizeof(float) * header2[2] * header2[1]);


				is.close();

				phi = new float* [(size_t)rows];

				tot_size += rows * sizeof(float*);

				for (int i = 0; i < rows; ++i) {
					phi[i] = new float[(size_t)cols];

					tot_size += cols * sizeof(float);
					for (int j = 0; j < cols; ++j) {
						phi[i][j] = array_[j * rows + i];
					}
				}
				delete[] array_;
			}
		public:
			Hash(const std::string &file) {
				loadIndex(file);
			}
			Hash(Preprocess& prep_, u_int32_t L_, u_int32_t K_, u_int32_t c_) {
				Parameter param_(prep_, L_, K_);
				Partition parti_(c_, prep_);
				parti = parti_;
				N = param_.N;
				dim = param_.dim;
				L = param_.L;
				K = param_.K;
				S = param_.S;
				tot_size = 6 * sizeof(int);
				load_funtable(std::string("/home/cty/mips-benchmark/algorithm/fargo/MyfunctionXTheta.data"));
				SetHash();
				GetHash(prep_);
				GetTables(prep_);
			}
			void SetHash() {
				hashpar.rndAs1 = new float* [S];
				hashpar.rndAs2 = new float* [S];

				tot_size += 1ll * S * sizeof(float*) * 2;
				
				hashpar.S = S;
				hashpar.dim = dim;
				tot_size += 2 * sizeof(int);

				for (int i = 0; i < S; i++) {
					hashpar.rndAs1[i] = new float[dim];
					hashpar.rndAs2[i] = new float[1];
					tot_size += (dim + 1) * sizeof(float);
				}

				std::mt19937 rng(int(std::time(0)));
				std::normal_distribution<float> nd;
				for (int j = 0; j < S; j++) {
					for (int i = 0; i < dim; i++) {
						hashpar.rndAs1[j][i] = (nd(rng));
					}
					for (int i = 0; i < 1; i++) {
						hashpar.rndAs2[j][i] = (nd(rng));
					}
				}
			}

			void GetHash(Preprocess& prep) {
				float* dataExpend = new float[N];
				tot_size += 1ll * N * sizeof(float);
				std::mt19937 rng(int(std::time(0)));
				std::uniform_real_distribution<float> ur(-1, 1);
				int count = 0;
				for (int j = 0; j < N; j++) {
					assert(parti.MaxLen[parti.chunks[j]] >= prep.SquareLen[j]);
					dataExpend[j] = sqrt(parti.MaxLen[parti.chunks[j]] - prep.SquareLen[j]);
					if (ur(rng) > 0) {
						dataExpend[j] *= -1;
						++count;
					}
					
				}
				hashval = new float* [N];
				for (int j = 0; j < N; j++) {
					hashval[j] = new float[S];
					for (int i = 0; i < S; i++) {
						hashval[j][i] = 1.0 - prep.fstdistfunc_(prep.data.val[j], hashpar.rndAs1[i], prep.dist_func_param_) + dataExpend[j] * hashpar.rndAs2[i][0];
					}
				}
				delete[] dataExpend;
			}
			void GetTables(Preprocess& prep) {
				int i, j, k;

				int num_bucket = 1 << K;

				myIndexes = new std::vector<int> * *[parti.num_chunk];
				myIndexes_size = new int ** [parti.num_chunk];
				tot_size += 1ll * parti.num_chunk * (sizeof(std::vector<int>**) + sizeof(int **));
				for (j = 0; j < parti.num_chunk; ++j) {
					myIndexes[j] = new std::vector<int> * [L];
					myIndexes_size[j] = new int * [L];
					tot_size += 1ll * L * (sizeof(std::vector<int>*) + sizeof(int *));
					for (i = 0; i < L; ++i) {
						myIndexes[j][i] = new std::vector<int>[num_bucket];
						myIndexes_size[j][i] = new int[num_bucket];
						for(int k = 0;k < num_bucket;k++) myIndexes_size[j][i][k] = 0;
						tot_size += 1ll * num_bucket * (sizeof(std::vector<int>) + sizeof(int));
					}
				}

				for (j = 0; j < L; j++) {
					for (i = 0; i < N; i++) {
						int start = j * K;
						int key = 0;
						for (k = 0; k < K; k++) {
							key = key << 1;
							if (this->hashval[i][ start + k] > 0) {
								++key;
							}
						}
						myIndexes[(size_t)parti.chunks[i]][j][key].push_back(i);
						myIndexes_size[(size_t)parti.chunks[i]][j][key]++;
						tot_size += sizeof(int);
					}
				}
			}
			~Hash() {
				for (int j = 0; j < parti.num_chunk; ++j) {
					for (int i = 0; i < L; ++i) {
						for (int l = 0; l < (1 << K); ++l) {
							std::vector<int>().swap(myIndexes[j][i][l]);
						}
						delete[] myIndexes[j][i];
						delete[] myIndexes_size[j][i];
					}
					delete[] myIndexes[j];
					delete[] myIndexes_size[j];
				}
				delete[] myIndexes;
				delete[] myIndexes_size;

				if(hashval != nullptr) clear_2d_array(hashval, N);
				clear_2d_array(hashpar.rndAs1, S);
				clear_2d_array(hashpar.rndAs2, S);
				clear_2d_array(phi, rows);
			}
		};

		struct Res { //the result of knns
			float inp;
			int id;
			Res() = default;
			Res(int id_, float inp_):id(id_), inp(inp_) {}
			bool operator < (const Res& rhs) const {
				return inp > rhs.inp;
			}
		};

		struct hash_pair {
			float val;
			int bias;

			bool operator < (const hash_pair& rhs) const {
				return val < rhs.val;
			}
		};

		struct indice_pair {
			int key;
			int end;
			int table_id;
			float score;
			bool operator < (const indice_pair& rhs) const {
				return score > rhs.score;
			}
		};

		struct mp_pair {
			int end;
			float score = FLT_MAX;
		};

		struct gmp_pair {
			int key;
			int table_id;
			float score;
			bool operator < (const gmp_pair& rhs) const {
				return score < rhs.score;
			}
		};

		class Query {
		private:
			// the parameter "c" in "c-ANN"
			float c;
			//which chunk is accessed
			int chunks;
			int UB = 100;
			float* query_point;

			// the hash value of query point
			float* hashval;

			std::vector<std::vector<hash_pair>> weigh;
			std::vector<float> total_score;

			float** mydata;
			int dim;

			float inp_LB;
			float break_ratio;
			//Percentage of compute nodes in total nodes

			std::vector<int> keys;

			void shift(indice_pair& ip0, indice_pair& res) {
				res = ip0;
				++res.end;
				res.key += weigh[res.table_id][res.end].bias - weigh[res.table_id][(size_t)res.end - 1].bias;
				res.score = ip0.score + weigh[res.table_id][res.end].val - weigh[res.table_id][(size_t)res.end - 1].val;
			}
			void expand(indice_pair& ip0, indice_pair& res) {
				res = ip0;
				++res.end;
				res.key += weigh[res.table_id][res.end].bias;
				res.score = ip0.score + weigh[res.table_id][res.end].val;
			}

			std::priority_queue<indice_pair> global_min;
			indice_pair* ProbingSequence;
			int SequenceLen;

			int tid = -1;

			float varphi(float x, float theta, Hash& hash) {
				int c0 = (int)floor((x - hash.smin) / hash.sstep);
				int r0 = (int)floor((theta - hash.tmin) / hash.tstep);
				return hash.phi[std::min(r0, hash.rows - 1)][std::min(c0, hash.cols - 1)];
			}

		public:
			// k-NN
			int k;
			// Indice of query point in dataset. Be equal to -1 if the query point isn't in the dataset.
			int flag;
			//
			float norm;
			//
			int cost = 0;
			//cost of each partition
			std::vector<int> costs;
			//
			float time_total = 0;
			//
			float time_hash = 0;
			//
			float time_sift = 0;

			float time_verify = 0;
			// query result:<indice of ANN,distance of ANN>
			std::vector<Res> res;

			void cal_hash(Hash& hash, Preprocess& prep) {
				query_point = prep.query.val[flag];
				norm = 1.0 - prep.fstdistfunc_(query_point, query_point, prep.dist_func_param_);
				norm = sqrt(norm);

				hashval = new float[hash.S];
				for (int i = 0; i < hash.S; ++i) {
					hashval[i] = (1 - prep.fstdistfunc_(query_point, hash.hashpar.rndAs1[i], prep.dist_func_param_)) / norm;
				}
				hash_pair hp0;
				this->weigh.resize(hash.L);
				total_score.resize(hash.L);
				for (int i = 0; i < hash.L; i++) {
					this->weigh[i].resize(hash.K);
					total_score[i] = 0;
					for (int j = 0; j < hash.K; j++) {
						hp0.val = abs(hashval[(size_t)(i * hash.K + j)]);

						hp0.bias = 1 << (hash.K - 1 - j);
						if (hashval[i * hash.K + j] > 0) {
							hp0.bias *= -1;
						}
						weigh[i][j] = hp0;
						total_score[i] += -hp0.val / 2;
					}
					std::sort(weigh[i].begin(), weigh[i].end());
				}
			}
			void siftF(Hash& hash, Preprocess& prep) {
				ProbingSequence = new indice_pair[(size_t)(hash.L * (size_t)(1 << hash.K) + 1)];
				SequenceLen = 0;

				indice_pair ip0, ip1;
				if (!global_min.empty()) {
					system("pause");
				}
				keys.resize(hash.L);
				for (int i = 0; i < hash.L; i++) {
					int key = 0;
					for (int j = 0; j < hash.K; j++) {
						key = key << 1;
						if (this->hashval[i * hash.K + j] > 0) {
							++key;
						}
					}
					keys[i] = key;
					ip0.key = key + weigh[i][0].bias;

					ip0.end = 0;
					ip0.table_id = i;
					ip0.score = weigh[i][0].val;
					global_min.push(ip0);
				}

				std::vector<bool> flag_(hash.N, false);
				Res res_PQ[10100] = {};
				int size = 0;
				inp_LB = MINFLOAT;
				costs.resize(hash.parti.num_chunk);

				for (int t = hash.parti.num_chunk - 1; t >= 0; t--) {
					if (sqrt(hash.parti.MaxLen[t]) * norm < inp_LB / c) break;
					if (hash.parti.nums[t] < 4 * CANDIDATES) {
						int num_cand = hash.parti.EachParti[t].size();
						for (int j = 0; j < num_cand; j++) {
							int& x = hash.parti.EachParti[t][j];
							res_PQ[size].id = x;
							res_PQ[size].inp = 1.0 - prep.fstdistfunc_(mydata[x], query_point, prep.dist_func_param_);
							costs[t]++;
							if (size < UB) {
								size++;
								std::push_heap(res_PQ, res_PQ + size);
							}
							else if(res_PQ[0].inp < res_PQ[size].inp) {
								size++;
								std::push_heap(res_PQ, res_PQ + size);
								std::pop_heap(res_PQ, res_PQ + size);
								size--;
							}
						}
					}
					else {
						chunks = t;
						knnF(res_PQ, hash, prep, hash.myIndexes[t], flag_, size);
					}
					
					if (size == UB) inp_LB = res_PQ[0].inp;
				}

				res.clear();

				int len = size;
				res.resize(len);
				int rr = len - 1;
				while (rr >= 0) {
					res[rr] = res_PQ[0];
					std::pop_heap(res_PQ, res_PQ + size);
					size--;
					rr--;
				}


				for (int i = 0; i < hash.parti.num_chunk; i++) {
					cost += costs[i];
				}
			}
			void knnF(Res* res_PQ, Hash& hash, Preprocess& prep, std::vector<int>** table, std::vector<bool>& flag_, int& size) {
				int cnt = 0;
				float inpK = -1.0f;
				if (size == UB) inpK = res_PQ[0].inp;
				float Max_inp = this->norm * sqrt(hash.parti.MaxLen[chunks]);

				for (int i = 0; i < hash.L; i++) {
					for (auto& x : table[i][keys[i]]) {
						if (flag_[x] == false){
							res_PQ[size].id = x;
							res_PQ[size].inp = 1.0 - prep.fstdistfunc_(mydata[x], query_point, prep.dist_func_param_);
							cnt++;
							costs[chunks]++;
							if (size < UB) {
								size++;
								std::push_heap(res_PQ, res_PQ + size);
							}
							else if (res_PQ[0].inp < res_PQ[size].inp) {
								size++;
								std::push_heap(res_PQ, res_PQ + size);
								std::pop_heap(res_PQ, res_PQ + size);
								size--;
								inpK = res_PQ[0].inp;
							}
							flag_[x] = true;
						}
					}
				}

				float Max_score = sqrt(2.0f / pi);
				float coeff = (pi / Max_score);
				int probingNum = 0;
				indice_pair ip0, ip1;

				int len = hash.K;
				float reduced_score, est_inp;

				int MaxNum = hash.parti.nums[chunks] * 1.0;
				float beta = 1.0f;

				while (cnt < (int)(beta * MaxNum)
					&& (probingNum < SequenceLen || (!global_min.empty()))
					) {
					if (probingNum < SequenceLen) {
						ip1 = ProbingSequence[probingNum];
						++probingNum;
					}
					else {
						ip1 = global_min.top();
						ProbingSequence[SequenceLen++] = ip1;
						++probingNum;

						global_min.pop();
						if (ip1.end < len - 1) {
							this->shift(ip1, ip0);
							global_min.push(ip0);

							this->expand(ip1, ip0);
							global_min.push(ip0);
						}
					}
					for (auto& x : table[ip1.table_id][ip1.key]) {
						if (flag_[x] == false) {
							res_PQ[size].id = x;
							res_PQ[size].inp = 1.0 - prep.fstdistfunc_(mydata[x], query_point, prep.dist_func_param_);
							cnt++;
							costs[chunks]++;
							if (size < UB) {
								size++;
								std::push_heap(res_PQ, res_PQ + size);
							}
							else if (res_PQ[0].inp < res_PQ[size].inp) {
								size++;
								std::push_heap(res_PQ, res_PQ + size);
								std::pop_heap(res_PQ, res_PQ + size);
								size--;
								inpK = res_PQ[0].inp;
							}
							flag_[x] = true;
						}
					}
					if (inpK > 0) {
						float theta = acos(std::min(0.9999f, inpK / (c * Max_inp)));
						float pr = varphi(ip1.score, theta, hash);
						if (pow(1 - pr, hash.L) < 1.1 - break_ratio) break;
					}

				}
				//endwhile
			}
		public:
			Query(int id, float c_, int k_, Hash& hash, Preprocess& prep, int ub_, float break_ratio_) {
				flag = id;
				c = c_;
				k = k_;
				UB = ub_;
				break_ratio = break_ratio_;
				mydata = prep.data.val;
				dim = prep.query.dim;
				cal_hash(hash, prep);
				siftF(hash, prep);
			}

			~Query() {
				delete[] ProbingSequence;
				delete[] hashval;
			}
		};
	}
	std::vector<std::vector<u_int32_t> > Alg0_mfalsh(mf_alsh::Hash& myslsh, float c_, int m_, int k_, int L_, int K_, Preprocess& prep, int ms_, float break_ratio_ = 1.0f) {
		std::vector<std::vector<u_int32_t> > res(prep.query.N);
		int Qnum = prep.query.N;
		#pragma omp parallel for
		for (int j = 0; j < Qnum; j++) {
			mf_alsh::Query query(j, c_, k_, myslsh, prep, m_, break_ratio_);
			#pragma omp critical
			{
				std::vector<u_int32_t> res_id(m_);
				std::transform(query.res.begin(), query.res.end(), res_id.begin(),
						[](const mf_alsh::Res& rs) { return rs.id; }
				);
				res_id.resize(k_);
				res[j].swap(res_id);
			}
		}
		return res;
	}
}