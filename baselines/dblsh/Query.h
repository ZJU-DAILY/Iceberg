#pragma once
#include "Preprocess.h"
#include "dblsh.h"
#include <vector>
#include <queue>
#include <set>


namespace dblsh {

class Query
{
private:
	// the parameter "c" in "c-ANN"
	float c;
	//which chunk is accessed
	float init_w = 1.0f;

	float* query_point;
	// the hash value of query point
	float* hashval;

	float** mydata;
	int dim;

public:
	// k-NN
	unsigned k;
	// Indice of query point in dataset. Be equal to -1 if the query point isn't in the dataset.
	unsigned flag;
	
	float beta = 0;

	unsigned cost = 0;

	//#access;
	int num_access = 0;
	//
	unsigned rounds = 0;
	//
	float time_total = 0;
	//
	float time_hash = 0;
	//
	float time_sift = 0;

	float time_verify = 0;
	// query result:<indice of ANN,distance of ANN>
	std::vector<Res> res;

	void cal_hash(Hash& hash, Preprocess& prep);
	void sift(Hash& hash, Preprocess& prep);
public:
	Query(unsigned id, float c_, unsigned k_, Hash& hash, Preprocess& prep, float beta);

	~Query();
};


Query::Query(unsigned id, float c_, unsigned k_, Hash& hash, Preprocess& prep, float beta_)
{
	flag = id;
	c = c_;
	k = k_;
	beta = beta_;

	init_w = hash.R_min * 4.0f * c_ * c_;

	mydata = prep.data.val;
	dim = prep.data.dim;
	query_point = prep.data.query[flag];

	cal_hash(hash, prep);

	sift(hash, prep);
}

void Query::cal_hash(Hash& hash, Preprocess& prep)
{
	hashval = new float[hash.S];
	for (int i = 0; i < hash.S; ++i) {
		
		hashval[i] = prep.fstdistfunc_(query_point, hash.hashpar.rndAs1[i], prep.dist_func_param_);
	}
}

void Query::sift(Hash& hash, Preprocess& prep)
{
	float t = 1.0f;
	if (hash.N < 70000) {
		t = 200.0;
	}
	else if (hash.N >= 70000 && hash.N < 500000) {
		//200-1001
		t = 200 + 7 * ((float)hash.N) / 10000;
		t = 1000;
	}
	else if (hash.N >= 500000 && hash.N < 2000000) {
		//501-1000
		t = 501.0+(1000.0-501.0)/150.0* ((float)hash.N) / 10000;
		t = 2000;
	}
	else if (hash.N >= 2000000 && hash.N < 2000000000) {
		//1000-20000
		t = 1000.0 + 19000.0 / 200000.0 * ((float)hash.N) / 10000;
		t = 20000;
	}
	else if (hash.N >= 2000000000) {
		//1000-20000
		t = 20000.0;
	}
	t *= 2;
	int T = (int)t * 2 * hash.L + k;

	if (prep.hasT) {
		T = beta * hash.N + k;
	}
	T = beta * hash.N + k;

 	Visitor* visits = new Visitor(hash.N, hash.K, hash.dim, k, T, mydata, query_point, hash.R_min * c);

	while (! visits->termination && rounds<=30) {
		rounds++;
		res.clear();
		
		for (int i = 0; i < hash.L; ++i) {
			int sta = i * hash.K;
			visits->q_mbr = new float[2 * visits->low_dim];
			for (int j = 0; j < visits->low_dim; ++j) {
				visits->q_mbr[2 * j] = hashval[j+sta] - init_w / 2;
				visits->q_mbr[2 * j + 1] = hashval[j+sta] + init_w / 2;
			}
			hash.myIndexes[i]->windows_query(visits);
		}
		init_w *= this->c;
	}

	cost = visits->count;
	num_access = visits->num_leaf_access + visits->num_nonleaf_access;
	num_access = visits->num_nonleaf_access;

	std::partial_sort(visits->res, visits->res + visits->k, visits->res + visits->count);
	res.assign(visits->res, visits->res + visits->k);
	delete visits;

}


Query::~Query()
{
	delete[] hashval;
	//
}


std::vector<std::vector<u_int32_t>> lshknn(float c, int k, Hash& myslsh, Preprocess& prep, float beta) {
	int Qnum = prep.data.Q_num;
	std::vector<std::vector<u_int32_t> > res(Qnum);
	#pragma omp parallel for
	for (unsigned j = 0; j < Qnum; j++)
	{
		Query query(j, c, k, myslsh, prep, beta);
		#pragma omp critical
		{
			std::vector<u_int32_t> res_id(k);
			std::transform(query.res.begin(), query.res.end(), res_id.begin(),
					[](const Res& rs) { return rs.id; }
			);
			res_id.resize(k);
			res[j].swap(res_id);
		}
	}
	return res;
}

}