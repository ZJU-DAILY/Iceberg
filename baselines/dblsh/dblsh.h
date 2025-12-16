#pragma once
#include "Preprocess.h"
#include "RStarTree.h"
#include "hnswlib.h"
#include <cmath>
#include <assert.h>
#include <unordered_map>
#include <vector>
#include <random>
#include <queue>

namespace dblsh {

class Hash
{
private:
	std::vector<TreeDataP<float>**> hs;
public:
	int N = 0;
	int dim = 0;
	// Number of hash functions
	int S = 0;
	int L = 0;
	int K = 0;
	int MaxSize = 0;
	float R_min = 1.0f;
	
	HashParam hashpar;
	RStarTree<TreeDataP, float>** myIndexes = NULL;

	Hash(Preprocess& prep_, Parameter& param_);
	void SetHash();
	void GetHash(Preprocess& prep);
	void GetTables(Preprocess& prep);
	bool IsBuilt(const std::string& file);
	~Hash();
};


Hash::Hash(Preprocess& prep_, Parameter& param_)
{
	N = param_.N;
	dim = param_.dim;
	L = param_.L;
	K = param_.K;
	S = L * K;
	R_min = param_.R_min;
	MaxSize = param_.MaxSize;
	SetHash();
	GetHash(prep_);
	GetTables(prep_);
}

bool Hash::IsBuilt(const std::string& file)
{
	return 0;
}

void Hash::SetHash()
{
	hashpar.rndAs1 = new float* [S];
	hashpar.rndAs2 = new float* [S];

	for (unsigned i = 0; i < S; i++) {
		hashpar.rndAs1[i] = new float[dim];
		hashpar.rndAs2[i] = new float[1];
	}

	std::mt19937 rng(unsigned(0));
	std::normal_distribution<float> nd;
	for (unsigned j = 0; j < S; j++)
	{
		for (unsigned i = 0; i < dim; i++)
		{
			hashpar.rndAs1[j][i] = (nd(rng));
		}
		for (unsigned i = 0; i < 1; i++)
		{
			hashpar.rndAs2[j][i] = (nd(rng));
		}
	}
}

void Hash::GetHash(Preprocess& prep)
{
	hs.resize(L);
	for (int i = 0; i < L; ++i) {
		hs[i] = new (TreeDataP<float>*[N]);
	}
	for (int i = 0; i < L; ++i) {
		for (int j = 0; j < N; ++j) {
			hs[i][j] = new TreeDataP<float>(K);
			hs[i][j]->id = j;
			for (int k = 0; k < K; ++k) {
				hs[i][j]->data[k] = prep.fstdistfunc_(prep.data.val[j], hashpar.rndAs1[i * K + k],
									prep.dist_func_param_);
			}
		}
	}
}

void Hash::GetTables(Preprocess& prep)
{

	int page_len = 4096;
	if (N < 70000) {
		page_len = 4096;
	}
	else if (N >= 70000 && N < 500000) {
		//200-501
		page_len = 8192;
	}
	else if (N >= 500000 && N < 2000000) {
		//501-1000
		page_len = 8192 * 1;
	}
	else if (N >= 2000000 && N < 2000000000) {
		//1000-20000
		page_len = 8192 * 2;
	}
	else if (N >= 2000000000) {
		//1000-20000
		page_len = 8192 * 2;
	}


	PageFile::c_policy pols[] = { PageFile::C_FULLMEM, PageFile::C_LRU, PageFile::C_MRU, PageFile::C_NO_CACHE };
	int pagefile_cache_size = 0; //we use full memory
	bool force_new = true; //if file exists, we will force overwrite it

	myIndexes = new RStarTree<TreeDataP, float>*[L];

	
	if (N > 1) {
		for (int i = 0; i < L; ++i) {
			std::string file = "RStar_index_file//" + std::to_string(i) + "_rstar.rt";
			myIndexes[i] = new RStarTree<TreeDataP, float>(file.c_str(), K, page_len, ".",
				pols[0], pagefile_cache_size, force_new);

			myIndexes[i]->bulkload_str(hs[i], N, 0.7);
			

			for (int j = 0; j < N; ++j) {
				delete[] hs[i][j]->data;
				hs[i][j]->data = NULL;
				delete hs[i][j];
			}
			delete[] hs[i];
			hs[i] = NULL;

		}
	}
	else {
		for (int i = 0; i < L; ++i) {
			std::string file = "RStar_index_file//" + std::to_string(i) + "_rstar.rt";
			myIndexes[i] = new RStarTree<TreeDataP, float>(file.c_str(), K, page_len, ".",
				pols[0], pagefile_cache_size, force_new);

			//insert one by one
			for (int j = 0; j < N; ++j) {
				myIndexes[i]->insert(hs[i][j]);
			}
			for (int j = 0; j < N; ++j) {
				delete[] hs[i][j]->data;
				hs[i][j]->data = NULL;
				delete hs[i][j];
			}
			delete[] hs[i];
			hs[i] = NULL;
		}
	}
}


Hash::~Hash()
{
	clear_2d_array(hashpar.rndAs1, S);
	clear_2d_array(hashpar.rndAs2, S);
	for (int i = 0; i < L; ++i) {
		delete myIndexes[i];
	}
	delete myIndexes;
}

}