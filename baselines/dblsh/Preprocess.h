#pragma once
#include <cmath>
#include <assert.h>
#include <unordered_map>
#include <algorithm>
#include <io.h>
#include "space_ip.h"
#include "space_l2.h"

namespace dblsh {

struct Data
{
	// Dimension of data
	unsigned dim = 0;
	// Number of data
	unsigned N = 0;
	// Number of query
	unsigned Q_num = 0;
	// Data matrix
	float** val = nullptr;
	float** query=nullptr; 
};

struct Ben
{
	unsigned N = 0;
	unsigned num = 0;
	int** indice = nullptr;
	float** dist = nullptr;
};

struct HashParam
{
	// the value of a in S hash functions
	float** rndAs1 = nullptr;
	// the value of a in S hash functions
	float** rndAs2 = nullptr;

};

template <class T>
void clear_2d_array(T** array, int n) {
    for (int i = 0; i < n; ++i) {
        delete[] array[i];
    }
    delete[] array;
}

class Preprocess
{
public:
	Data data;
	float* SquareLen = NULL;
	float** Dists = NULL;
	bool hasT = false;
	float beta = 0.1f;
	DISTFUNC<float> fstdistfunc_;
	void* dist_func_param_{nullptr};

	Preprocess(benchmark::DataSet& dataset_, benchmark::QuerySet& queryset_, dblsh::SpaceInterface<float>* s) {
		data.dim = dataset_._dimension;
		data.N = dataset_._num;
		data.Q_num = queryset_._num;
		data.val = new float* [data.N];
		fstdistfunc_ = s->get_dist_func();
		dist_func_param_ = s->get_dist_func_param();
		for (int i = 0; i < data.N; i++) {
			data.val[i] = dataset_._vecs[i].data();
		}
		data.query = new float* [data.Q_num];
		for (int i = 0; i < data.Q_num; i++) {
			data.query[i] = queryset_._queries[i]._vec.data();
		}
	}
	~Preprocess() {
		delete[] SquareLen;
	}
};

class Parameter //N,dim,S, L, K, M, W;
{
public:
	unsigned N = 0;
	unsigned dim = 0;
	// Number of hash functions
	unsigned S = 0;
	//#L Tables; 
	unsigned L = 0;
	// Dimension of the hash table
	unsigned K = 0;
	int MaxSize = 0;
	float R_min = 1.0f;
	Parameter(Preprocess& prep, unsigned L_, unsigned K_, float rmin_) {
		N = prep.data.N;
		dim = prep.data.dim;
		L = L_;
		K = K_;
		MaxSize = 5;
		R_min = rmin_;
	}
	~Parameter(){};
};

}