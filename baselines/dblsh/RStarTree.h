#ifndef _R_STAR_TREE_H_
#define _R_STAR_TREE_H_

#include <cstring>
#include <vector>
#include <deque>
#include <map>
#include <unordered_map> //C++11
#include <limits>
#include <queue>
using namespace std;

//#define _USE_HASH_NODE_BUFFER_
//only dependency
#include "PageFile.h"
#include "space_l2.h"
#include "hnswlib.h"
//fix for max macro clash with numeric_limits max, min and also std max, min templates
#undef max
#undef min

namespace dblsh {

inline 
bool Is_Intersect(float* mbr, float* data, const int& dim)
{
	if (dim == 10) {
		return !((*(mbr)> (*data))|| (*(++mbr) < (*data))
			|| (*(++mbr) > *(++data)) || (*(++mbr) < (*data))
			|| (*(++mbr) > *(++data)) || (*(++mbr) < (*data))
			|| (*(++mbr) > *(++data)) || (*(++mbr) < (*data))
			|| (*(++mbr) > *(++data)) || (*(++mbr) < (*data))
			|| (*(++mbr) > *(++data)) || (*(++mbr) < (*data))
			|| (*(++mbr) > *(++data)) || (*(++mbr) < (*data))
			|| (*(++mbr) > *(++data)) || (*(++mbr) < (*data))
			|| (*(++mbr) > *(++data)) || (*(++mbr) < (*data))
			|| (*(++mbr) > *(++data)) || (*(++mbr) < (*data))
			);
	}

	for (int i = 0; i < dim; ++i) {
		if ((*(mbr++) > (*data)) ||
			(*(mbr++) < (*(data++)))) {
			return false;
		}
	}
	return true;
}

inline
bool Is_enclosing(float* qmbr, float* nmbr, const int& dim)
{
	if (dim == 10) {
		return !(
			(*(qmbr) > *(nmbr)) || (*(++qmbr) < *(++nmbr))
			|| (*(++qmbr) > *(++nmbr)) || (*(++qmbr) < *(++nmbr))
			|| (*(++qmbr) > *(++nmbr)) || (*(++qmbr) < *(++nmbr))
			|| (*(++qmbr) > *(++nmbr)) || (*(++qmbr) < *(++nmbr))
			|| (*(++qmbr) > *(++nmbr)) || (*(++qmbr) < *(++nmbr))
			|| (*(++qmbr) > *(++nmbr)) || (*(++qmbr) < *(++nmbr))
			|| (*(++qmbr) > *(++nmbr)) || (*(++qmbr) < *(++nmbr))
			|| (*(++qmbr) > *(++nmbr)) || (*(++qmbr) < *(++nmbr))
			|| (*(++qmbr) > *(++nmbr)) || (*(++qmbr) < *(++nmbr))
			|| (*(++qmbr) > *(++nmbr)) || (*(++qmbr) < *(++nmbr))
			);
	}

	for (int i = 0; i < dim; ++i) {
		if (
			(*(qmbr++) > *(nmbr++)) ||
			(*(qmbr++) < *(nmbr++))
			) {
			return false;
		}
	}
	return true;
}

struct Res//the result of knns
{
	int id = -1;
	float dist = FLT_MAX;
	bool operator < (const Res& rhs) const {
		return dist < rhs.dist;
	}
};

class Visitor
{
public:
	int* flag;
	float** original_data;
	int count;
	int num_leaf_access;
	int num_nonleaf_access;
	int low_dim;
	int data_dim;
	int k;
	int MAXCOST;
	float* q_mbr;
	float* q_point;
	float Radius;//search radius
	bool termination;//stop if it is true
	float kth_dist;
	Res* res;

	Visitor(int N, int K_, int dim_, int k_,int MaxC, float** mydata, float* query, float R_min) {
		flag = new int[N];
		for (int i = 0; i < N; ++i) {
			flag[i] = true;
		}
		count = 0;
		num_leaf_access = 0;
		num_nonleaf_access = 0;
		termination = false;
		low_dim = K_;
		data_dim = dim_;
		k = k_;
		MAXCOST = MaxC;
		original_data = mydata;
		q_point = query;
		q_mbr = NULL;
		Radius = R_min;
		kth_dist = FLT_MAX;
		res = new Res[MAXCOST];
	}
	~Visitor() {
		delete[] q_mbr;
		delete[] flag;
		delete[] res;
	}
};


//data structure declaration
enum INSERT_RESULT {INSERT_SPLIT, INSERT_REINSERT, INSERT_NONE};
enum INTERSECT_RESULT {INTERSECT_OVERLAP, INTERSECT_INSIDE, INTERSECT_NONE};

//helper data structures
//remember we need ids to force same sort order
template <typename T = float>
struct mbr_compare_less
{
	T **all_mbr; // the array of all MBRs
	int *all_id; // the id of every entry, we need this to ensure same order under unstable sort
	int comp_dim; // the specific dimension we need to compare
	int lu_tag; // be 0 - lower, 1 - upper

	mbr_compare_less(T **mbrs, int *ids, int the_dim, int tag) : all_mbr(mbrs), all_id(ids), comp_dim(the_dim), lu_tag(tag) {}
	bool operator()(const int id1, const int id2) const
	{
		T val1=all_mbr[id1][2*comp_dim+lu_tag];
		T val2=all_mbr[id2][2*comp_dim+lu_tag];
		if(val1<val2) return true;
		else if((val1==val2)&&(all_id[id1]<all_id[id2])) return true;
		else return false;
	}
};

template <typename T = float>
struct mbr_compare_center_less
{
	T **all_mbr; // the array of all MBRs
	int *all_id; // the id of every entry, we need this to ensure same order under unstable sort
	T *center; // the center of all
	int comp_dim; // the specific dimension we need to compare

	mbr_compare_center_less(T **mbrs, int *ids, T *the_center, int the_dim) : all_mbr(mbrs), all_id(ids), center(the_center), comp_dim(the_dim) {}
	bool operator()(const int id1, const int id2) const
	{
		T val1=(T)0;
		T val2=(T)0;
		T d;
		for(int i=0;i<2*comp_dim;i+=2)
		{
			d=(all_mbr[id1][i]+all_mbr[id1][i+1])/(T)2-center[i/2];
			val1+=d*d;
			d=(all_mbr[id2][i]+all_mbr[id2][i+1])/(T)2-center[i/2];
			val2+=d*d;
		}

		if(val1<val2) return true;
		else if((val1==val2)&&(all_id[id1]<all_id[id2])) return true;
		else return false;
	}
};

//calculate margin of MBR
template <typename T>
T margin(int dim, T *mbr)
{
	T sum=(T)0;
	for(int i=0;i<2*dim;i+=2) sum+=mbr[i+1]-mbr[i];
	return sum;
}

//calculate volume of MBR
template <typename T>
double area(int dim, T *mbr)
{
	double sum=(double)1;
	for(int i=0;i<2*dim;i+=2) sum*=mbr[i+1]-mbr[i];
	return sum;
}

//calculate overlap of MBR
template <typename T>
double overlap(int dim, T *r1, T *r2)
{
	double sum=(double)1;
	for(int i=0;i<2*dim;i+=2)
	{
		T lower=max<T>(r1[i], r2[i]);
		T upper=min<T>(r1[i+1], r2[i+1]);
		if(lower>=upper) return (double)0; //do not overlap
		else sum*=upper-lower;
	}

	return sum;
}

//calculate the enlargement of r1 with r2 and store the result in mbr_buf
template <typename T>
void enlarge(int dim, T *mbr_buf, T *r1, T *r2)
{
	for(int i=0;i<2*dim;i+=2)
	{
		mbr_buf[i]=min<T>(r1[i], r2[i]);
		mbr_buf[i+1]=max<T>(r1[i+1], r2[i+1]);
	}
}

//for data sorting in bulkloading
template <template <typename TP> class Data, typename T>
struct data_compare_center_less
{
	Data<T> **all_data; // the array of all data, with one field indicating the ids
	int comp_dim; // the specific dimension we need to compare

	data_compare_center_less(Data<T> **data_array, int the_dim) : all_data(data_array), comp_dim(the_dim) {}
	bool operator()(const int id1, const int id2) const
	{
		T val1=all_data[id1]->get_center_dim(comp_dim);
		T val2=all_data[id2]->get_center_dim(comp_dim);
		if(val1<val2) return true;
		else if((val1==val2)&&(all_data[id1]->id<all_data[id2]->id)) return true;
		else return false;
	}
};

//TreeDataP provides the T type point data
//T is better to be int, float or double, default is float
template <typename T = float>
class TreeDataP
{
public:
	int dim; // the dimension
	T *data; // contain a list of T
	int id; // for identification purpose
	T dist; // anything to give a number

	//constructor
	TreeDataP(int dimension) : dim(dimension), data(NULL), id(-1), dist((T)0)
	{
		data=new T[dim];
	}

	//destructor
	~TreeDataP()
	{
		if(data!=NULL) delete[] data;

		data = NULL;
	}

	//copy constructor and equal operator
	TreeDataP(const TreeDataP &other)
	{
		data=new T[other.dim];
		*this=other;
	}

	TreeDataP &	operator = (const TreeDataP &other)
	{
		if(&other!=this)
		{
			dim=other.dim;
			memcpy(data, other.data, dim*sizeof(T));
			id=other.id;
			dist=other.dist;
		}

		return (*this);
	}

	//fill mbr in the buffer
	T *get_mbr(T *buf)
	{
		for(int i=0;i<dim;i++) buf[2*i+1]=buf[2*i]=data[i];
		return buf;
	}

	//get area
	T get_area() { return (T)0; } //for point it is always 0

	//read from buffer, return next position
	const char *read_from_buffer(const char *buffer)
	{
		memcpy(data, buffer, dim*sizeof(T));
		buffer+=dim*sizeof(T);
		memcpy(&id, buffer, sizeof(int));
		buffer+=sizeof(int);
		memcpy(&dist, buffer, sizeof(T));
		return (buffer+=sizeof(T));
	}

	//write to buffer, return next position
	char *write_to_buffer(char *buffer)
	{
		memcpy(buffer, data, dim*sizeof(T));
		buffer+=dim*sizeof(T);
		memcpy(buffer, &id, sizeof(int));
		buffer+=sizeof(int);
		memcpy(buffer, &dist, sizeof(T));
		return (buffer+=sizeof(T));
	}

	//for bulkloading
	T get_center_dim(int dim)
	{
		return data[dim];
	}

	//return the total buffer size of this class
	static int get_size(int dim) { return dim*sizeof(T)+sizeof(int)+sizeof(T); }

	//point
	void print()
	{
		cout <<"[("<<data[0];
		for(int i=1;i<dim;i++) cout <<data[i]<<", ";
		cout <<"), "<<id<<", "<<dist<<"]";
		cout.flush();
	}
};

//TreeDataR provides the T type MBR data
//T is better to be int, float or double, default is float
template <typename T = float>
class TreeDataR
{
public:
	int dim; // the dimension
	T *data; // contain a list of T
	int id; // for identification purpose
	T dist; // anything to give a number

	//constructor
	TreeDataR(int dimension)  : dim(dimension), data(NULL), id(-1), dist((T)0)
	{
		data=new T[2*dim];
	}

	//destructor
	~TreeDataR()
	{
		if(data!=NULL) delete[] data;
	}

	//copy constructor and equal operator
	TreeDataR(const TreeDataR &other)
	{
		data=new T[2*other.dim];
		*this=other;
	}

	TreeDataR &	operator = (const TreeDataR &other)
	{
		if(&other!=this)
		{
			dim=other.dim;
			memcpy(data, other.data, 2*dim*sizeof(T));
			id=other.id;
			dist=other.dist;
		}

		return (*this);
	}

	//fill mbr in the buffer
	T *get_mbr(T *buf)
	{
		for(int i=0;i<2*dim;i++) buf[i]=data[i];
		return buf;
	}

	//get area
	T get_area()
	{
		T res=(T)1;
		for(int i=0;i<2*dim;i+=2)
		{
			res*=data[i+1]-data[i];
		}
		return res;
	}

	//read from buffer, return next position
	const char *read_from_buffer(const char *buffer)
	{
		memcpy(data, buffer, 2*dim*sizeof(T));
		buffer+=2*dim*sizeof(T);
		memcpy(&id, buffer, sizeof(int));
		buffer+=sizeof(int);
		memcpy(&dist, buffer, sizeof(T));
		return (buffer+=sizeof(T));
	}

	//write to buffer, return next position
	char *write_to_buffer(char *buffer)
	{
		memcpy(buffer, data, 2*dim*sizeof(T));
		buffer+=2*dim*sizeof(T);
		memcpy(buffer, &id, sizeof(int));
		buffer+=sizeof(int);
		memcpy(buffer, &dist, sizeof(T));
		return (buffer+=sizeof(T));
	}

	//for bulkloading
	T get_center_dim(int dim)
	{
		return data[2*dim]+(data[2*dim+1]-data[2*dim])/2;
	}

	//return the total buffer size of this class
	static int get_size(int dim) { return 2*dim*sizeof(T)+sizeof(int)+sizeof(T); }

	//point
	void print()
	{
		cout <<"[("<<data[0];
		for(int i=1;i<dim;i++) cout <<data[i]<<", ";
		cout <<"), "<<id<<", "<<dist<<"]";
		cout.flush();
	}
};

//default template argument declaration
template <template <typename TP> class Data = TreeDataP, typename T = float> class RStarTree;
template <template <typename TP> class Data = TreeDataP, typename T = float> class RSTNode;
template <template <typename TP> class Data = TreeDataP, typename T = float> class RSTNonLeafNode;
template <template <typename TP> class Data = TreeDataP, typename T = float> class RSTLeafNode;

template <template <typename TP> class Data, typename T>
class RSTNode
{
public:
	RStarTree<Data, T> *the_tree; // the link to whole tree
	int dim; // the dimension
	int capacity; // the maximum possible number of entries
	int entry_num; // the actual number of entries
	int page_id; // the corresponding page id
	int depth; // the depth is actually 1-byte in disk, we use int for alignment
	bool dirty; // if it has been modified

	RSTNode(RStarTree<Data, T> *rt);
	virtual ~RSTNode();
	virtual void flush() = 0; //flush this to disk
	virtual void windows_query(Visitor* visit) = 0;
	//split is not virtual
	//according to mbr, then return the split point
	//and fill in split_buf with all indexes
	//the split_buf must be pre-allocated having at least entry_num size
	int split(T **mbr_array, int *mbr_id, int *split_buf);

	//virtual interface
	virtual int get_num_of_data() = 0; //return the total number of object in this sub-tree rooted at this node
	virtual Data<T> *get(int index, Data<T> *data_buf) = 0; //returns the i-th object in this sub-tree rooted at this node

	virtual void read_from_buffer(const char *buffer) = 0; //reads data from buffer, return next position
	virtual void write_to_buffer(char *buffer) = 0; //writes data to buffer, return next position

	virtual bool is_leaf_node() = 0; //judge if it is a leaf node or not

	virtual T *get_mbr(T *buf) = 0; //returns mbr enclosing whole page
	virtual void print() = 0; //prints the information

	virtual INSERT_RESULT insert(Data<T> *d, RSTNode<Data, T> **sn) = 0; //insert recrsively
};

template <template <typename TP> class Data, typename T>
class RSTNonLeafNode : public RSTNode<Data, T>
{
public:
	bool child_is_leaf; // if all its children are leaf nodes or not
	T **entry_mbr; // the MBR of every child entry
	int *entry_page; // the page id of every child entry
	int *entry_num_data; // the number of data for every entry child
	RSTNode<Data, T> **children; // the actual pointer to children

	RSTNonLeafNode(RStarTree<Data, T> *rt); //generate new page
	RSTNonLeafNode(RStarTree<Data, T> *rt, int page); //load fron existing page
	~RSTNonLeafNode();
	void flush(); //flush this to disk
	void windows_query(Visitor* visit);
	//new functions
	static int get_capacity(int page_len, int dim);

	void enter(T *new_mbr, int new_page, int new_num_data, RSTNode<Data, T> *child); //enter-in one data
	int choose_subtree(T *mbr); //choose optimal sub tree to insert
	INTERSECT_RESULT intersect_subtree(int sub_id, T *mbr); //test the relationshiop between MBR of subtree and a given MBR
	RSTNode<Data, T> *get_child(int sub_id); //get the actual child node and store it

	//overriding
	void split(RSTNonLeafNode<Data, T> *sn); //split and put splitted nodes into sn

	//virtual function implementations
	int get_num_of_data(); //return the total number of object in this sub-tree rooted at this node
	Data<T> *get(int index, Data<T> *data_buf); //returns the i-th object in this sub-tree rooted at this node

	void read_from_buffer(const char *buffer); //reads data from buffer, return next position
	void write_to_buffer(char *buffer); //writes data to buffer, return next position

	bool is_leaf_node() { return false; } //always return false because it is not a leaf node

	T *get_mbr(T *buf); //returns mbr enclosing whole page
	void print(); //prints the information

	INSERT_RESULT insert(Data<T> *d, RSTNode<Data, T> **sn); //insert recrsively
};

template <template <typename TP> class Data, typename T>
class RSTLeafNode : public RSTNode<Data, T>
{
private:
	
public:
	T* nmbr = NULL;
	T maxLengthOfSide = 0;
	Data<T> **data; // the contained data

	RSTLeafNode(RStarTree<Data, T> *rt); //generate new page
	RSTLeafNode(RStarTree<Data, T> *rt, int page); //load fron existing page
	~RSTLeafNode();
	void flush(); //flush this to disk
	void windows_query(Visitor* visit);

	//new function
	static int get_capacity(int page_len, int dim);

	//overriding
	void split(RSTLeafNode<Data, T> *sn); //split and put splitted nodes into sn

	//virtual function implementations
	int get_num_of_data(); //return the total number of object in this sub-tree rooted at this node
	Data<T> *get(int index, Data<T> *data_buf); //returns the i-th object in this sub-tree rooted at this node

	void read_from_buffer(const char *buffer); //reads data from buffer, return next position
	void write_to_buffer(char *buffer); //writes data to buffer, return next position

	bool is_leaf_node() { return true; } //always return true because it is actually a leaf node

	T *get_mbr(T *buf); //returns mbr enclosing whole page
	void print(); //prints the information

	INSERT_RESULT insert(Data<T> *d, RSTNode<Data, T> **sn); //insert recrsively
};

template <template <typename TP> class Data, typename T>
class RStarTree
{
public:
	int dim; // the dimensionaltiy
	int num_data; // the number of data
	int num_leaf; // the number of leaves
	int num_nonleaf; // the number of non-leaves
	bool root_is_leaf; // whether root is already a leaf node
	int root_page; // the page id of the root

	PageFile *file; // the associated page based memory/file
	char *header; // the buffer for header
	RSTNode<Data, T> *root; // the root node

	int leaf_capacity; //the leaf capacity
	int nonleaf_capacity; //the non-leaf capacity

	bool *reinsrt_lvl; // whether reinsert occur on that level
	list<Data<T> > *reinsrt_cands; // the candidates for re-insertion
	void windows_query(Visitor* visit);

#ifndef _USE_HASH_NODE_BUFFER_
	map<int, RSTNode<Data, T> *> *node_buf; // provide a buffer mechanism
#else
	unordered_map<int, RSTNode<Data, T> *> *node_buf; // provide a buffer mechanism
#endif

	//create a new tree or modify existing tree
	RStarTree(const char *fname, int dimension, int p_len, const char *path=NULL,
		PageFile::c_policy pol=PageFile::C_FULLMEM, int c_size=0, bool force_new=false);
	virtual ~RStarTree();

	void load_root(); //load root node
	const char *read_header(const char *buffer); //load the r*-tree header, return next position
	char *write_header(char *buffer); //write the r*-tree header, return next position

	Data<T> *get(int index, Data<T> *data_buf); //get i-th data
	void insert(Data<T> *the_data); //insert one data
	void bulkload_str(Data<T> **the_data_array, int num_data, double load_factor); //bulk load all data, using Sort-Tile-Recursive method
};

template <template <typename TP> class Data, typename T>
void RStarTree<Data, T>::windows_query(Visitor* visit)
{
	//we make sure root node is loaded
	load_root();

	root->windows_query(visit);
}

//then we start implemenatation

//RSTNode
template <template <typename TP> class Data, typename T>
RSTNode<Data, T>::RSTNode(RStarTree<Data, T> *rt)
	: the_tree(rt), dim(rt->dim), capacity(0), entry_num(0), page_id(-1), depth(-1), dirty(false)
{
}

template <template <typename TP> class Data, typename T>
RSTNode<Data, T>::~RSTNode()
{
}

template <template <typename TP> class Data, typename T>
int RSTNode<Data, T>::split(T **mbr_array, int *mbr_id, int *split_buf)
{
	//since nodes must be filled at least 40%
	//we calculate a limit
	int limit=(int)((double)entry_num*0.4);

	//we build helper
	int *sort_lower=new int[entry_num];
	int *sort_upper=new int[entry_num];
	T *mbr1=new T[2*dim];
	T *mbr2=new T[2*dim];

	//find best split axis
	int split_axis=-1;
	T min_margin=numeric_limits<T>::max();
	for(int i=0;i<dim;i++) //for every dimension
	{
		//make the order as natural order 0,1,2,3,...,entry_num-1
		for(int j=0;j<entry_num;j++) sort_lower[j]=sort_upper[j]=j;

		//sort by lower and upper value in dim i
		std::sort(sort_lower, sort_lower+entry_num, mbr_compare_less<T>(mbr_array, mbr_id, i, 0)); //lower
		std::sort(sort_upper, sort_upper+entry_num, mbr_compare_less<T>(mbr_array, mbr_id, i, 1)); //upper

		T the_margin=(T)0;
		int l;

		//for every possible divide of sort_lower
		for(int k=0;k<entry_num-2*limit+1;k++)
		{
			//first we calculate the margin of R1
			for(int s=0;s<2*dim;s+=2)
			{
				mbr1[s]=numeric_limits<T>::max();
				mbr1[s+1]=-numeric_limits<T>::max();
			}
			for(l=0;l<limit+k;l++)
			{
				for(int s=0;s<2*dim;s+=2)
				{
					mbr1[s]=min<T>(mbr1[s], mbr_array[sort_lower[l]][s]);
					mbr1[s+1]=max<T>(mbr1[s+1], mbr_array[sort_lower[l]][s+1]);
				}
			}
			the_margin+=margin<T>(dim, mbr1);

			//then we calculate the margin of R2
			for(int s=0;s<2*dim;s+=2)
			{
				mbr1[s]=numeric_limits<T>::max();
				mbr1[s+1]=-numeric_limits<T>::max();
			}
			for(;l<entry_num;l++)
			{
				for(int s=0;s<2*dim;s+=2)
				{
					mbr1[s]=min<T>(mbr1[s], mbr_array[sort_lower[l]][s]);
					mbr1[s+1]=max<T>(mbr1[s+1], mbr_array[sort_lower[l]][s+1]);
				}
			}
			the_margin+=margin<T>(dim, mbr1);
		}

		//for every possible divide of sort_upper
		for(int k=0;k<entry_num-2*limit+1;k++)
		{
			//first we calculate the margin of R1
			for(int s=0;s<2*dim;s+=2)
			{
				mbr1[s]=numeric_limits<T>::max();
				mbr1[s+1]=-numeric_limits<T>::max();
			}
			for(l=0;l<limit+k;l++)
			{
				for(int s=0;s<2*dim;s+=2)
				{
					mbr1[s]=min<T>(mbr1[s], mbr_array[sort_upper[l]][s]);
					mbr1[s+1]=max<T>(mbr1[s+1], mbr_array[sort_upper[l]][s+1]);
				}
			}
			the_margin+=margin<T>(dim, mbr1);

			//then we calculate the margin of R2
			for(int s=0;s<2*dim;s+=2)
			{
				mbr1[s]=numeric_limits<T>::max();
				mbr1[s+1]=-numeric_limits<T>::max();
			}
			for(;l<entry_num;l++)
			{
				for(int s=0;s<2*dim;s+=2)
				{
					mbr1[s]=min<T>(mbr1[s], mbr_array[sort_upper[l]][s]);
					mbr1[s+1]=max<T>(mbr1[s+1], mbr_array[sort_upper[l]][s+1]);
				}
			}
			the_margin+=margin<T>(dim, mbr1);
		}

		//judge optimum
		if(the_margin<min_margin)
		{
			split_axis=i;
			min_margin=the_margin;
		}
	}

	//we use best split axis
	//make the order as natural order 0,1,2,3,...,entry_num-1
	for(int j=0;j<entry_num;j++) sort_lower[j]=sort_upper[j]=j;

	//sort by lower and upper value in split_axis
	std::sort(sort_lower, sort_lower+entry_num, mbr_compare_less<T>(mbr_array, mbr_id, split_axis, 0)); //lower
	std::sort(sort_upper, sort_upper+entry_num, mbr_compare_less<T>(mbr_array, mbr_id, split_axis, 1)); //upper

	//find best split using overlap and dead area
	double min_over=numeric_limits<double>::max();
	double min_dead=numeric_limits<double>::max();
	int split_point=-1;
	bool use_lower_sort=false;

	//for every possible divide of sort_lower and sort_upper
	for(int k=0;k<entry_num-2*limit+1;k++)
	{
		double the_dead;
		double the_over;
		int l;

		//for lower_sort
		//first we calculate R1
		the_dead=(T)0;
		for(int s=0;s<2*dim;s+=2)
		{
			mbr1[s]=numeric_limits<T>::max();
			mbr1[s+1]=-numeric_limits<T>::max();
		}
		for(l=0;l<limit+k;l++)
		{
			for(int s=0;s<2*dim;s+=2)
			{
				mbr1[s]=min<T>(mbr1[s], mbr_array[sort_lower[l]][s]);
				mbr1[s+1]=max<T>(mbr1[s+1], mbr_array[sort_lower[l]][s+1]);
			}
			the_dead-=area<T>(dim, mbr_array[sort_lower[l]]);
		}
		the_dead+=area<T>(dim, mbr1);

		//then we calculate R2
		for(int s=0;s<2*dim;s+=2)
		{
			mbr2[s]=numeric_limits<T>::max();
			mbr2[s+1]=-numeric_limits<T>::max();
		}
		for(;l<entry_num;l++)
		{
			for(int s=0;s<2*dim;s+=2)
			{
				mbr2[s]=min<T>(mbr2[s], mbr_array[sort_lower[l]][s]);
				mbr2[s+1]=max<T>(mbr2[s+1], mbr_array[sort_lower[l]][s+1]);
			}
			the_dead-=area<T>(dim, mbr_array[sort_lower[l]]);
		}
		the_dead+=area<T>(dim, mbr2);
		the_over=overlap<T>(dim, mbr1, mbr2);

		if((the_over<min_over)||((the_over==min_over)&&(the_dead<min_dead)))
		{
			min_over=the_over;
			min_dead=the_dead;
			split_point=limit+k;
			use_lower_sort=true;
		}

		//for upper_sort
		//first we calculate R1
		the_dead=(T)0;
		for(int s=0;s<2*dim;s+=2)
		{
			mbr1[s]=numeric_limits<T>::max();
			mbr1[s+1]=-numeric_limits<T>::max();
		}
		for(l=0;l<limit+k;l++)
		{
			for(int s=0;s<2*dim;s+=2)
			{
				mbr1[s]=min<T>(mbr1[s], mbr_array[sort_upper[l]][s]);
				mbr1[s+1]=max<T>(mbr1[s+1], mbr_array[sort_upper[l]][s+1]);
			}
			the_dead-=area<T>(dim, mbr_array[sort_upper[l]]);
		}
		the_dead+=area<T>(dim, mbr1);

		//then we calculate R2
		for(int s=0;s<2*dim;s+=2)
		{
			mbr2[s]=numeric_limits<T>::max();
			mbr2[s+1]=-numeric_limits<T>::max();
		}
		for(;l<entry_num;l++)
		{
			for(int s=0;s<2*dim;s+=2)
			{
				mbr2[s]=min<T>(mbr2[s], mbr_array[sort_upper[l]][s]);
				mbr2[s+1]=max<T>(mbr2[s+1], mbr_array[sort_upper[l]][s+1]);
			}
			the_dead-=area<T>(dim, mbr_array[sort_upper[l]]);
		}
		the_dead+=area<T>(dim, mbr2);
		the_over=overlap<T>(dim, mbr1, mbr2);

		if((the_over<min_over)||((the_over==min_over)&&(the_dead<min_dead)))
		{
			min_over=the_over;
			min_dead=the_dead;
			split_point=limit+k;
			use_lower_sort=false;
		}
	}

	//return the split
	if(use_lower_sort) memcpy(split_buf, sort_lower, entry_num*sizeof(int));
	else memcpy(split_buf, sort_upper, entry_num*sizeof(int));

	delete[] sort_lower;
	delete[] sort_upper;
	delete[] mbr1;
	delete[] mbr2;

	return split_point;
}

//RSTNonLeafNode
template <template <typename TP> class Data, typename T>
RSTNonLeafNode<Data, T>::RSTNonLeafNode(RStarTree<Data, T> *rt) : RSTNode<Data, T>(rt), child_is_leaf(true)
{
	int page_len=rt->file->page_len;

	//compute capacity
	this->capacity=get_capacity(page_len, this->dim);

	//create entries
	entry_mbr=new (T *[this->capacity]);
	for(int i=0;i<this->capacity;i++) entry_mbr[i]=new T[2*this->dim];
	entry_page=new int[this->capacity];
	memset(entry_page, 0xff, this->capacity*sizeof(int));
	entry_num_data=new int[this->capacity];
	memset(entry_num_data, 0, this->capacity*sizeof(int));
	children=new (RSTNode<Data, T> *[this->capacity]);
	for(int i=0;i<this->capacity;i++) children[i]=NULL;

	//append the block to disk file
	char *buf_page=new char[page_len];
	memset(buf_page, 0, page_len);
	this->page_id=rt->file->append_page(buf_page);
	delete[] buf_page;

	//add to buffer
	if(rt->node_buf->count(this->page_id)>0)
	{
		//if already exist a node with same page, then this is a bug!!!
		printf("Fatal error in file: %s, func: %s, line: %d\n", __FILE__, __FUNCTION__, __LINE__);
		printf("Program halt!\n");
		exit(-1);
	}
	else (*(rt->node_buf))[this->page_id]=this;

	rt->num_nonleaf++;

	this->dirty=true;
}

template <template <typename TP> class Data, typename T>
RSTNonLeafNode<Data, T>::RSTNonLeafNode(RStarTree<Data, T> *rt, int page) : RSTNode<Data, T>(rt)
{
	int page_len=rt->file->page_len;

	//compute capacity
	this->capacity=get_capacity(page_len, this->dim);

	//create entries
	entry_mbr=new (T *[this->capacity]);
	for(int i=0;i<this->capacity;i++) entry_mbr[i]=new T[2*this->dim];
	entry_page=new int[this->capacity];
	memset(entry_page, 0xff, this->capacity*sizeof(int));
	entry_num_data=new int[this->capacity];
	memset(entry_num_data, 0, this->capacity*sizeof(int));
	children=new (RSTNode<Data, T> *[this->capacity]);
	for(int i=0;i<this->capacity;i++) children[i]=NULL;

	//read disk block
	this->page_id=page;
	const char *buf_page=rt->file->read_page(page);
	read_from_buffer(buf_page);

	//add to buffer
	if(rt->node_buf->count(this->page_id)>0)
	{
		//if already exist a node with same page, then this is a bug!!!
		printf("Fatal error in file: %s, func: %s, line: %d\n", __FILE__, __FUNCTION__, __LINE__);
		printf("Program halt!\n");
		exit(-1);
	}
	else (*(rt->node_buf))[this->page_id]=this;

	this->dirty=false;
}

template <template <typename TP> class Data, typename T>
RSTNonLeafNode<Data, T>::~RSTNonLeafNode()
{
	flush();

	if(entry_mbr!=NULL)
	{
		for(int i=0;i<this->capacity;i++) delete[] entry_mbr[i];
		delete[] entry_mbr;
	}
	if(entry_page!=NULL) delete[] entry_page;
	if(entry_num_data!=NULL) delete[] entry_num_data;
	if(children!=NULL)
	{
		for(int i=0;i<this->capacity;i++) if(children[i]!=NULL) delete children[i];
		delete[] children;
	}

	//remove from buffer
	this->the_tree->node_buf->erase(this->page_id);
}

template <template <typename TP> class Data, typename T>
void RSTNonLeafNode<Data, T>::flush()
{
	if(this->dirty)
	{
		char *buf_page=new char[this->the_tree->file->page_len];
		memset(buf_page, 0, this->the_tree->file->page_len);
		write_to_buffer(buf_page);
		this->the_tree->file->write_page(buf_page, this->page_id);
		delete[] buf_page;
	}
}

template <template <typename TP> class Data, typename T>
int RSTNonLeafNode<Data, T>::get_capacity(int page_len, int dim)
{
	//compute header size and entry size
	int header_size=sizeof(bool)+sizeof(char)+sizeof(int);
	int entry_size=2*dim*sizeof(T)+sizeof(int)+sizeof(int);

	//compute capacity
	return (page_len - header_size) / entry_size - 1;
}

template <template <typename TP> class Data, typename T>
void RSTNonLeafNode<Data, T>::enter(T *new_mbr, int new_page, int new_num_data, RSTNode<Data, T> *child)
{
	//judge whether space is enough
	if(this->entry_num>=this->capacity)
	{
		printf("Fatal error in file: %s, func: %s, line: %d\n", __FILE__, __FUNCTION__, __LINE__);
		printf("Program halt!\n");
		exit(-1);
	}

	//copy to vacancy
	memcpy(entry_mbr[this->entry_num], new_mbr, 2*this->dim*sizeof(T));
	entry_page[this->entry_num]=new_page;
	entry_num_data[this->entry_num]=new_num_data;
	children[this->entry_num]=child;

	this->entry_num++;
	//should be some release work
}

template <template <typename TP> class Data, typename T>
int RSTNonLeafNode<Data, T>::choose_subtree(T *mbr)
{
	int result_id = -1;
	T *mbr_buf=new T[2*this->dim];

	//first we test the relationship between given mbr and the subtrees
	//test if mbr is inside some subtree or not
	int inside_count=0;
	vector<int> inside_id;
	for(int i=0;i<this->entry_num;i++)
	{
		if(intersect_subtree(i, mbr)==INTERSECT_INSIDE)
		{
			inside_id.push_back(i);
			inside_count++;
		}
	}

	if(inside_count==1)
	{
		//only one subtree found
		result_id=inside_id.front();

		if (result_id < 0) {
			printf("Fatal error in file: %s, func: %s, line: %d\n", __FILE__, __FUNCTION__, __LINE__);
			printf("Program halt!\n");
			system("pause");
			exit(-1);
		}
	}
	else if(inside_count>1)
	{
		//multiple subtree found
		//we will choose the one with lowest surface area
		double min_area=numeric_limits<double>::max();
		for(vector<int>::iterator iter=inside_id.begin();iter!=inside_id.end();++iter)
		{
			double the_area=area<T>(this->dim, entry_mbr[*iter]);
			if(the_area<min_area)
			{
				result_id=*iter;
				min_area=the_area;
			}
		}

		if (result_id < 0) {
			printf("Fatal error in file: %s, func: %s, line: %d\n", __FILE__, __FUNCTION__, __LINE__);
			printf("Program halt!\n");
			system("pause");
			exit(-1);
		}
	}
	else
	{
		//no subtree found!
		//we distinguish two cases
		if(child_is_leaf)
		{
			//for leaf, we use the subtree which will overlap the least with its siblings after insert data to it
			//when there is a overlap tie, we use the subtree with the least enlargement
			//when there is still a enlargement tie, we use the subtree with the least area
			double min_overlap=numeric_limits<double>::max();
			double min_enlarge=numeric_limits<double>::max();
			double min_area=numeric_limits<double>::max();

			for(int i=0;i<this->entry_num;i++)
			{
				//enlarge the i-th entry
				enlarge(this->dim, mbr_buf, mbr, entry_mbr[i]);

				//calculate the area and enlargement
				double the_area=area(this->dim, entry_mbr[i]);
				double the_enlarge=area(this->dim, mbr_buf)-the_area;

				//calculate the delta overlap before and after enlargement
				double before_overlap=(double)0;
				double after_overlap=(double)0;
				for(int j=0;j<this->entry_num;j++)
				{
					if(j!=i)
					{
						before_overlap+=overlap(this->dim, entry_mbr[i], entry_mbr[j]);
						after_overlap+=overlap(this->dim, mbr_buf, entry_mbr[j]);
					}
				}
				after_overlap-=before_overlap;

				//judge optimal
				if((after_overlap<min_overlap)||
					((after_overlap==min_overlap)&&(the_enlarge<min_enlarge))||
					((after_overlap==min_overlap)&&(the_enlarge==min_enlarge)&&(the_area<min_area)))
				{
					result_id=i;
					min_overlap=after_overlap;
					min_enlarge=the_enlarge;
					min_area=the_area;
				}
			}

			if (result_id < 0) {
				printf("Fatal error in file: %s, func: %s, line: %d\n", __FILE__, __FUNCTION__, __LINE__);
				printf("Program halt!\n");
				system("pause");
				exit(-1);
			}
		}
		else
		{
			//for nonleaf, we use the subtree which will be enlarged the least
			//when there is a enlargement tie, we use the subtree with the least area
			double min_enlarge=numeric_limits<double>::max();
			double min_area=numeric_limits<double>::max();

			for(int i=0;i<this->entry_num;i++)
			{
				//enlarge the i-th entry
				enlarge(this->dim, mbr_buf, mbr, entry_mbr[i]);

				//calculate the area and enlargement
				double the_area=area(this->dim, entry_mbr[i]);
				double the_enlarge=area(this->dim, mbr_buf)-the_area;

				//judge optimal
				if((the_enlarge<min_enlarge)||
					((the_enlarge==min_enlarge)&&(the_area<min_area)))
				{
					result_id=i;
					min_enlarge=the_enlarge;
					min_area=the_area;
				}
			}

			if (result_id < 0) {
				printf("Fatal error in file: %s, func: %s, line: %d\n", __FILE__, __FUNCTION__, __LINE__);
				printf("Program halt!\n");
				system("pause");
				exit(-1);
			}
		}
		
		//update the mbr
		enlarge(this->dim, mbr_buf, mbr, entry_mbr[result_id]);
		memcpy(entry_mbr[result_id], mbr_buf, 2*this->dim*sizeof(T));
	}

	if (result_id < 0) {
		printf("Fatal error in file: %s, func: %s, line: %d\n", __FILE__, __FUNCTION__, __LINE__);
		printf("Program halt!\n");
		system("pause");
		exit(-1);
	}

	//release
	delete[] mbr_buf;

	return result_id;
}

template <template <typename TP> class Data, typename T>
INTERSECT_RESULT RSTNonLeafNode<Data, T>::intersect_subtree(int sub_id, T *mbr)
{
	bool is_overlap=true;
	bool is_inside=true;

	for(int i=0;i<2*this->dim;i+=2)
	{
		if(is_overlap&&
			((mbr[i]>entry_mbr[sub_id][i+1])||
			(mbr[i+1]<entry_mbr[sub_id][i])))
			is_overlap=false;
		if(is_inside&&
			((mbr[i]<entry_mbr[sub_id][i])||
			(mbr[i+1]>entry_mbr[sub_id][i+1])))
			is_inside=false;
	}

	if(is_inside) return INTERSECT_INSIDE;
	else if(is_overlap) return INTERSECT_OVERLAP;
	else return INTERSECT_NONE;
}

template <template <typename TP> class Data, typename T>
RSTNode<Data, T> *RSTNonLeafNode<Data, T>::get_child(int sub_id)
{
	if(children[sub_id]==NULL)
	{
#ifndef _USE_HASH_NODE_BUFFER_
		typename map<int, RSTNode<Data, T> *>::iterator iter=this->the_tree->node_buf->find(entry_page[sub_id]);
#else
		typename unordered_map<int, RSTNode<Data, T> *>::iterator iter=this->the_tree->node_buf->find(entry_page[sub_id]);
#endif		
		if(iter!=this->the_tree->node_buf->end())
		{
			//fetch from buffer
			children[sub_id]=iter->second;
		}
		else
		{
			if(child_is_leaf) children[sub_id]=new RSTLeafNode<Data, T>(this->the_tree, entry_page[sub_id]);
			else children[sub_id]=new RSTNonLeafNode<Data, T>(this->the_tree, entry_page[sub_id]);
		}
	}

	return children[sub_id];
}

template <template <typename TP> class Data, typename T>
void RSTNonLeafNode<Data, T>::split(RSTNonLeafNode<Data, T> *sn)
{
	int *split_buf=new int[this->entry_num];

	//do split
	int dist=RSTNode<Data, T>::split(entry_mbr, entry_page, split_buf);

	//allocate temporary memory
	T **mbr_temp=new (T *[this->entry_num]);
	int *page_temp=new int[this->entry_num];
	int *num_data_temp=new int[this->entry_num];
	RSTNode<Data, T> **children_temp=new (RSTNode<Data, T> *[this->entry_num]);

	//assign the value to temp
	for(int i=0;i<this->entry_num;i++)
	{
		mbr_temp[i]=entry_mbr[split_buf[i]];
		page_temp[i]=entry_page[split_buf[i]];
		num_data_temp[i]=entry_num_data[split_buf[i]];
		children_temp[i]=children[split_buf[i]];
	}

	//first we all thing back to original
	//this is quivalent to a re-ordering to original
	memcpy(entry_mbr, mbr_temp, this->entry_num*sizeof(T*));
	memcpy(entry_page, page_temp, this->entry_num*sizeof(int));
	memcpy(entry_num_data, num_data_temp, this->entry_num*sizeof(int));
	memcpy(children, children_temp, this->entry_num*sizeof(RSTNode<Data, T> *));

	//we have filled original with 0~dist new entry
	//then we exchange the dist~entry_num with the sn
	//because we need unused buffer from sn to make sure all mbr links in original are usable
	for(int i=dist;i<this->entry_num;i++)
	{
		//swap entry_mbr[dist~n] with sn->entry_mbr[0~n-dist]
		T *temp=entry_mbr[i];
		entry_mbr[i]=sn->entry_mbr[i-dist];
		sn->entry_mbr[i-dist]=temp;

		sn->entry_page[i-dist]=entry_page[i];
		entry_page[i]=-1;
		sn->entry_num_data[i-dist]=entry_num_data[i];
		entry_num_data[i]=0;

		//swap children[dist~n] with sn->children[0~n-dist]
		RSTNode<Data, T> *temp1=children[i];
		children[i]=sn->children[i-dist];
		sn->children[i-dist]=temp1;
	}

	//release
	delete[] split_buf;
	delete[] mbr_temp;
	delete[] page_temp;
	delete[] num_data_temp;
	delete[] children_temp;

	//update entry num
	sn->entry_num=this->entry_num-dist;
	this->entry_num=dist;
}

template <template <typename TP> class Data, typename T>
int RSTNonLeafNode<Data, T>::get_num_of_data()
{
	int sum=0;

	for(int i=0;i<this->entry_num;i++)
	{
		sum+=entry_num_data[i];
	}

	return sum;
}

template <template <typename TP> class Data, typename T>
Data<T> *RSTNonLeafNode<Data, T>::get(int index, Data<T> *data_buf)
{
	int sum=0;

	for(int i=0;i<this->entry_num;i++)
	{
		sum+=entry_num_data[i];
		if(sum>index)
		{
			RSTNode<Data, T> *sn=get_child(i);
			return get(index+entry_num_data[i]-sum, data_buf);
		}
	}

	return NULL;
}

template <template <typename TP> class Data, typename T>
void RSTNonLeafNode<Data, T>::read_from_buffer(const char *buffer)
{
	memcpy(&child_is_leaf, buffer, sizeof(bool));
	buffer+=sizeof(bool);
	this->depth=0; //clear high-24bits since we only read lower 8 bits
	memcpy(&this->depth, buffer, sizeof(char));
	buffer+=sizeof(char);
	memcpy(&this->entry_num, buffer, sizeof(int));
	buffer+=sizeof(int);
	for(int i=0;i<this->entry_num;i++)
	{
		memcpy(entry_mbr[i], buffer, 2*this->dim*sizeof(T));
		buffer+=2*this->dim*sizeof(T);
		memcpy(entry_page+i, buffer, sizeof(int));
		buffer+=sizeof(int);
		memcpy(entry_num_data+i, buffer, sizeof(int));
		buffer+=sizeof(int);
	}
}

template <template <typename TP> class Data, typename T>
void RSTNonLeafNode<Data, T>::windows_query(Visitor* visit)
{
	//++visit->num_nonleaf_access;
	for (int sub_id = 0; sub_id < this->entry_num; ++sub_id) {
		if (visit->termination) return;
		bool flag = true;
		++visit->num_nonleaf_access;
		for (int i = 0; i < 2 * this->dim; i += 2)
		{
			//++visit->num_nonleaf_access;
			if ((visit->q_mbr[i] > entry_mbr[sub_id][i + 1]) ||
				(visit->q_mbr[i + 1] < entry_mbr[sub_id][i])) {
				flag = false;
				break;
			}
			
		}

		if (flag) this->children[sub_id]->windows_query(visit);
	}
	
}

template <template <typename TP> class Data, typename T>
void RSTNonLeafNode<Data, T>::write_to_buffer(char *buffer)
{
	memcpy(buffer, &child_is_leaf, sizeof(bool));
	buffer+=sizeof(bool);
	memcpy(buffer, &this->depth, sizeof(char));
	buffer+=sizeof(char);
	memcpy(buffer, &this->entry_num, sizeof(int));
	buffer+=sizeof(int);
	for(int i=0;i<this->entry_num;i++)
	{
		memcpy(buffer, entry_mbr[i], 2*this->dim*sizeof(T));
		buffer+=2*this->dim*sizeof(T);
		memcpy(buffer, entry_page+i, sizeof(int));
		buffer+=sizeof(int);
		memcpy(buffer, entry_num_data+i, sizeof(int));
		buffer+=sizeof(int);
	}
}

template <template <typename TP> class Data, typename T>
T *RSTNonLeafNode<Data, T>::get_mbr(T *buf)
{
	memcpy(buf, entry_mbr[0], 2*this->dim*sizeof(T));

	for(int i=1;i<this->entry_num;i++)
	{
		for(int j=0;j<2*this->dim;j+=2)
		{
			buf[j]=min<T>(buf[j], entry_mbr[i][j]);
			buf[j+1]=max<T>(buf[j+1], entry_mbr[i][j+1]);
		}
	}

	return buf;
}

template <template <typename TP> class Data, typename T>
void RSTNonLeafNode<Data, T>::print()
{
	cout <<"[ depth:"<<this->depth;
	cout <<" ("<<entry_mbr[0][0]<<"~"<<entry_mbr[0][1];
	for(int j=2;j<2*this->dim;j+=2)
	{
		cout <<", "<<entry_mbr[0][j]<<"~"<<entry_mbr[0][j+1];
	}
	cout <<")";
	for(int i=1;i<this->entry_num;i++)
	{
		cout <<", ("<<entry_mbr[i][0]<<"~"<<entry_mbr[i][1];
		for(int j=2;j<2*this->dim;j+=2)
		{
			cout <<", "<<entry_mbr[i][j]<<"~"<<entry_mbr[i][j+1];
		}
		cout <<")";
	}
	cout <<"]";
	cout.flush();
}

template <template <typename TP> class Data, typename T>
INSERT_RESULT RSTNonLeafNode<Data, T>::insert(Data<T> *d, RSTNode<Data, T> **sn)
{
	//find optimal subtree
	T *mbr=new T[2*this->dim];
	int sub_id=choose_subtree(d->get_mbr(mbr)); //already updated entry_mbr

	//find sub tree node
	RSTNode<Data, T> *sub_tree=get_child(sub_id);

	//insert data into subtree
	RSTNode<Data, T> *new_sub=NULL;
	INSERT_RESULT ret=sub_tree->insert(d, &new_sub);

	if(ret!=INSERT_NONE)
	{
		//we have to update the mbr of sub_tree
		sub_tree->get_mbr(mbr);
		memcpy(entry_mbr[sub_id], mbr, 2*this->dim*sizeof(T));
	}

	//update number of data
	entry_num_data[sub_id]=sub_tree->get_num_of_data();

	if(ret==INSERT_SPLIT)
	{
		if(this->entry_num==this->capacity)
		{
			printf("Fatal error in file: %s, func: %s, line: %d\n", __FILE__, __FUNCTION__, __LINE__);
			printf("Program halt!\n");
			exit(-1);
		}

		//insert new entry
		new_sub->get_mbr(mbr);
		enter(mbr, new_sub->page_id, new_sub->get_num_of_data(), new_sub);

		//----------this is a modification to original code
		//we will test if this is correct
		if(this->entry_num==(this->capacity-1)) //back to original
		{
			//if the node is nearly full
			RSTNonLeafNode<Data, T> *sibling=new RSTNonLeafNode<Data, T>(this->the_tree); //append a new non-leaf node
			sibling->child_is_leaf=child_is_leaf;
			sibling->depth=this->depth;
			split(sibling);

			*sn=sibling; //pass it upwards
			ret=INSERT_SPLIT;
		}
		else ret=INSERT_NONE;
	}

	this->dirty=true;

	//release
	delete[] mbr;

	return ret;
}

//RSTLeafNode
template <template <typename TP> class Data, typename T>
RSTLeafNode<Data, T>::RSTLeafNode(RStarTree<Data, T> *rt) : RSTNode<Data, T>(rt)
{
	int page_len=rt->file->page_len;

	//every leaf node has a depth of 0
	this->depth=0;

	//compute capacity
	this->capacity=get_capacity(page_len, this->dim);

	//create entries
	data=new (Data<T> *[this->capacity]);
	for(int i=0;i<this->capacity;i++) data[i]=NULL;

	//append the block to disk file
	char *buf_page=new char[page_len];
	memset(buf_page, 0, page_len);
	this->page_id=rt->file->append_page(buf_page);
	delete[] buf_page;

	//add to buffer
	if(rt->node_buf->count(this->page_id)>0)
	{
		//if already exist a node with same page, then this is a bug!!!
		printf("Fatal error in file: %s, func: %s, line: %d\n", __FILE__, __FUNCTION__, __LINE__);
		printf("Program halt!\n");
		exit(-1);
	}
	else (*(rt->node_buf))[this->page_id]=this;

	rt->num_leaf++;

	this->dirty=true;
}

template <template <typename TP> class Data, typename T>
RSTLeafNode<Data, T>::RSTLeafNode(RStarTree<Data, T> *rt, int page) : RSTNode<Data, T>(rt)
{
	int page_len=rt->file->page_len;

	//every leaf node has a depth of 0
	this->depth=0;

	//compute capacity
	this->capacity=get_capacity(page_len, this->dim);

	//create entries
	data=new (Data<T> *[this->capacity]);
	for(int i=0;i<this->capacity;i++) data[i]=NULL;

	//read disk block
	this->page_id=page;
	const char *buf_page=rt->file->read_page(page);
	read_from_buffer(buf_page);

	//add to buffer
	if(rt->node_buf->count(this->page_id)>0)
	{
		//if already exist a node with same page, then this is a bug!!!
		printf("Fatal error in file: %s, func: %s, line: %d\n", __FILE__, __FUNCTION__, __LINE__);
		printf("Program halt!\n");
		exit(-1);
	}
	else (*(rt->node_buf))[this->page_id]=this;

	this->dirty=false;
}

template <template <typename TP> class Data, typename T>
RSTLeafNode<Data, T>::~RSTLeafNode()
{
	flush();

	if(data!=NULL)
	{
		for(int i=0;i<this->capacity;i++) if(data[i]!=NULL) delete data[i];
		delete[] data;
	}

	//remove from buffer
	this->the_tree->node_buf->erase(this->page_id);
}

template <template <typename TP> class Data, typename T>
void RSTLeafNode<Data, T>::flush()
{
	if(this->dirty)
	{
		char *buf_page=new char[this->the_tree->file->page_len];
		memset(buf_page, 0, this->the_tree->file->page_len);
		write_to_buffer(buf_page);
		this->the_tree->file->write_page(buf_page, this->page_id);
		delete[] buf_page;
	}
}

template <template <typename TP> class Data, typename T>
void RSTLeafNode<Data, T>::windows_query(Visitor* visit)
{
	if (visit->termination) return;
	
	SpaceInterface<float>* space = new L2Space(visit->data_dim);
	DISTFUNC<float> dist_ = space->get_dist_func();
	void* dist_param = space->get_dist_func_param();
	bool flag_enclosing = (visit->q_mbr[1] - visit->q_mbr[0] >= maxLengthOfSide) && Is_enclosing(visit->q_mbr, nmbr, this->dim);
	//delete mbr;
	for (int j = 0; j < this->entry_num; ++j) {
		++visit->num_leaf_access;

		if (
			flag_enclosing||
			Is_Intersect(visit->q_mbr, this->data[j]->data, this->dim)) {
			int data_id = this->data[j]->id;
			if (visit->flag[data_id]) {
				visit->flag[data_id] = false;
				visit->res[visit->count].id = data_id;
				visit->res[visit->count].dist = dist_(visit->original_data[data_id], visit->q_point, dist_param);
				
				
				if (visit->count == visit->k) {
					std::sort(visit->res, visit->res + visit->k);
					visit->kth_dist = visit->res[visit->k - 1].dist;
				}
				else if (visit->count > visit->k) {
					visit->kth_dist = visit->kth_dist < visit->res[visit->count - 1].dist ? visit->kth_dist : visit->res[visit->count - 1].dist;
				}

				
				++visit->count;
				if (visit->count == visit->MAXCOST) {
					visit->termination = true;
					delete space;
					return;
				}
			}
		}
	}
	delete space;
}

template <template <typename TP> class Data, typename T>
int RSTLeafNode<Data, T>::get_capacity(int page_len, int dim)
{
	//compute header size and entry size
	int header_size=sizeof(char)+sizeof(int);
	int entry_size=Data<T>::get_size(dim);

	//compute capacity
	return (page_len-header_size)/entry_size;
}

template <template <typename TP> class Data, typename T>
void RSTLeafNode<Data, T>::split(RSTLeafNode<Data, T> *sn)
{
	int *split_buf=new int[this->entry_num];
	T **entry_mbr=new (T*[this->entry_num]);
	int *entry_id=new int[this->entry_num];
	for(int i=0;i<this->entry_num;i++)
	{
		T *mbr_buf=new T[2*this->dim];
		entry_mbr[i]=data[i]->get_mbr(mbr_buf);
		entry_id[i]=data[i]->id;
	}

	//do split
	int dist=RSTNode<Data, T>::split(entry_mbr, entry_id, split_buf);

	//allocate temporary memory
	Data<T> **data_temp=new (Data<T> *[this->entry_num]);

	//assign the value to temp
	for(int i=0;i<this->entry_num;i++)
	{
		data_temp[i]=data[split_buf[i]];
	}

	//first we all thing back to original
	//this is quivalent to a re-ordering to original
	memcpy(data, data_temp, this->entry_num*sizeof(Data<T> *));

	//we have filled original with 0~dist new entry
	//then we exchange the dist~entry_num with the sn
	//because we need unused buffer from sn to make sure all mbr links in original are usable
	for(int i=dist;i<this->entry_num;i++)
	{
		//swap data[dist~n] with sn->data[0~n-dist]
		Data<T> *temp=data[i];
		data[i]=sn->data[i-dist];
		sn->data[i-dist]=temp;
	}

	//release
	delete[] split_buf;
	delete[] data_temp;
	for(int i=0;i<this->entry_num;i++) delete[] entry_mbr[i];
	delete[] entry_mbr;
	delete[] entry_id;

	//update entry num
	sn->entry_num=this->entry_num-dist;
	this->entry_num=dist;

	
}

template <template <typename TP> class Data, typename T>
int RSTLeafNode<Data, T>::get_num_of_data()
{
	return this->entry_num;
}

template <template <typename TP> class Data, typename T>
Data<T> *RSTLeafNode<Data, T>::get(int index, Data<T> *data_buf)
{
	if((index<0)||(index>=this->entry_num)) return NULL;

	*data_buf=*(data[index]);

	return data_buf;
}

template <template <typename TP> class Data, typename T>
void RSTLeafNode<Data, T>::read_from_buffer(const char *buffer)
{
	this->depth=0; //clear high-24bits since we only read lower 8 bits
	memcpy(&this->depth, buffer, sizeof(char));
	buffer+=sizeof(char);

	//we do a check here to detect potential bugs in the program
	if(this->depth!=0)
	{
		printf("Fatal error in file: %s, func: %s, line: %d\n", __FILE__, __FUNCTION__, __LINE__);
		printf("Program halt!\n");
		exit(-1);
	}

	memcpy(&this->entry_num, buffer, sizeof(int));
	buffer+=sizeof(int);
	for(int i=0;i<this->entry_num;i++)
	{
		data[i]=new Data<T>(this->dim);
		buffer=data[i]->read_from_buffer(buffer);
	}
}

template <template <typename TP> class Data, typename T>
void RSTLeafNode<Data, T>::write_to_buffer(char *buffer)
{
	this->depth=0;
	memcpy(buffer, &this->depth, sizeof(char));
	buffer+=sizeof(char);
	memcpy(buffer, &this->entry_num, sizeof(int));
	buffer+=sizeof(int);
	for(int i=0;i<this->entry_num;i++)
	{
		buffer=data[i]->write_to_buffer(buffer);
	}
}

template <template <typename TP> class Data, typename T>
T *RSTLeafNode<Data, T>::get_mbr(T *buf)
{
	T *mbr_buf=new T[2*this->dim];

	data[0]->get_mbr(buf);

	for(int i=1;i<this->entry_num;i++)
	{
		data[i]->get_mbr(mbr_buf);
		for(int j=0;j<2*this->dim;j+=2)
		{
			buf[j]=min<T>(buf[j], mbr_buf[j]);
			buf[j+1]=max<T>(buf[j+1], mbr_buf[j+1]);
			maxLengthOfSide = max<T>(buf[j + 1] - buf[j], maxLengthOfSide);
		}
	}

	delete[] mbr_buf;
	return buf;

	//return mbr_buf;
}

template <template <typename TP> class Data, typename T>
void RSTLeafNode<Data, T>::print()
{
	cout <<"{ depth:"<<this->depth;
	cout <<" ";
	data[0]->print();
	for(int i=1;i<this->entry_num;i++)
	{
		cout <<", ";
		data[i]->print();
	}
	cout <<"}";
	cout.flush();
}

template <template <typename TP> class Data, typename T>
INSERT_RESULT RSTLeafNode<Data, T>::insert(Data<T> *d, RSTNode<Data, T> **sn)
{
	if(this->entry_num==this->capacity)
	{
		printf("Fatal error in file: %s, func: %s, line: %d\n", __FILE__, __FUNCTION__, __LINE__);
		printf("Program halt!\n");
		exit(-1);
	}

	//fill data in
	data[this->entry_num]=new Data<T>(*d);
	this->entry_num++;

	this->dirty=true;

	//----------this is a modification to original code
	//we will test if this is correct
	if(this->entry_num==(this->capacity-1)) //back to original
	{
		if(this->the_tree->reinsrt_lvl[0]==false)
		{
			//if no re-insert on level 0
			//find center of the node
			T *center=new T[this->dim];
			nmbr = new T[2 * this->dim];
			get_mbr(nmbr);
			for(int i=0;i<2*this->dim;i+=2) center[i/2]=(nmbr[i]+nmbr[i+1])/((T)2);

			//build all MBRs
			int *sort_center=new int[this->entry_num];
			T **entry_mbr=new (T *[this->entry_num]);
			int *entry_id=new int[this->entry_num];
			for(int i=0;i<this->entry_num;i++)
			{
				sort_center[i]=i;
				T *mbr_buf=new T[2*this->dim];
				entry_mbr[i]=data[i]->get_mbr(mbr_buf);
				entry_id[i]=data[i]->id;
			}

			//sort by the distance of each center to the overall center
			std::sort(sort_center, sort_center+this->entry_num, mbr_compare_center_less<T>(entry_mbr, entry_id, center, this->dim)); //lower

			//we will keep 70% data
			int last_cand=(int)((double)this->entry_num*0.3);

			//allocate temporary memory
			int j;
			Data<T> **data_temp=new (Data<T> *[this->entry_num-last_cand]);

			//copy 70% data to temp
			for(j=0;j<this->entry_num-last_cand;j++)
			{
				data_temp[j]=data[sort_center[j]];
			}

			//insert last 30% to reinsertion list
			for(;j<this->entry_num;j++)
			{
				this->the_tree->reinsrt_cands->push_front(*(data[sort_center[j]]));
				delete data[sort_center[j]];
			}

			//copy temp back
			memcpy(data, data_temp, (this->entry_num-last_cand)*sizeof(Data<T> *));
			for(j=this->entry_num-last_cand;j<this->entry_num;j++) data[j]=NULL;

			//release
			//delete[] mbr;
			delete[] center;
			delete[] sort_center;
			for(int i=0;i<this->entry_num;i++) delete[] entry_mbr[i];
			delete[] entry_mbr;
			delete[] entry_id;
			delete[] data_temp;

			//update reinsert level
			this->the_tree->reinsrt_lvl[0]=true;

			//update size
			this->entry_num-=last_cand;

			this->dirty=true;

			get_mbr(nmbr);
			return INSERT_REINSERT;
		}
		else
		{
			*sn=new RSTLeafNode<Data, T>(this->the_tree); //append a new leaf node
			(*sn)->depth=this->depth;
			split((RSTLeafNode<Data, T> *)*sn);
			//sn->nmbr== new T[2 * this->dim];

			RSTLeafNode<Data, T>* ptr = static_cast<RSTLeafNode<Data, T> *>(*sn);
			ptr->nmbr = new T[2 * this->dim];
			ptr->get_mbr(ptr->nmbr);
			get_mbr(nmbr);

			return INSERT_SPLIT;
		}
	}
	else return INSERT_NONE;
}

//RStarTree
template <template <typename TP> class Data, typename T>
RStarTree<Data, T>::RStarTree(const char *fname, int dimension, int p_len, const char *path, PageFile::c_policy pol, int c_size, bool force_new)
	: root(NULL), reinsrt_lvl(NULL), reinsrt_cands(NULL), leaf_capacity(-1), nonleaf_capacity(-1)
{
	//open file
	file=new PageFile(fname, p_len, path, pol, c_size, force_new);
	header=new char[file->get_header_size()];

#ifndef _USE_HASH_NODE_BUFFER_
	node_buf=new map<int, RSTNode<Data, T> *>();
#else
	node_buf=new unordered_map<int, RSTNode<Data, T> *>();
#endif

	memset(header, 0, file->get_header_size());

	//allocate one data structure keeps using on insert function
	reinsrt_cands=new list<Data<T> >();

	leaf_capacity=RSTLeafNode<Data, T>::get_capacity(p_len, dimension);
	nonleaf_capacity=RSTNonLeafNode<Data, T>::get_capacity(p_len, dimension);

	if(file->new_file)
	{
		//new file
		dim=dimension;
		num_data=0;
		num_leaf=0;
		num_nonleaf=0;
		root_is_leaf=true;

		root=new RSTLeafNode<Data, T>(this);
		root_page=root->page_id;
	}
	else
	{
		//not new
		//read the header
		memcpy(header, file->read_header(), file->get_header_size());

		read_header(header);

		if(num_data==0)
		{
			//no data, then we create root
			root=new RSTLeafNode<Data, T>(this);
			root_page=root->page_id;
		}
		else load_root();
	}
}

template <template <typename TP> class Data, typename T>
RStarTree<Data, T>::~RStarTree()
{
	write_header(header);

	file->set_header(header);
	delete[] header;

	if(root!=NULL) delete root; //this will trigger all node flush

	if(reinsrt_lvl!=NULL) delete[] reinsrt_lvl;
	if(reinsrt_cands!=NULL) delete reinsrt_cands;

	if(!node_buf->empty())
	{
#ifndef _USE_HASH_NODE_BUFFER_
		for(typename map<int, RSTNode<Data, T> *>::iterator iter=node_buf->begin(); iter!=node_buf->end(); ++iter)
#else
		for(typename unordered_map<int, RSTNode<Data, T> *>::iterator iter=node_buf->begin(); iter!=node_buf->end(); ++iter)
#endif
		{
			delete iter->second;
		}
	}
	delete node_buf;

	delete file;

	//printf("saved R*-Tree containing %d non-leaf, %d leaf nodes and %d data\n", num_nonleaf, num_leaf, num_data);
}

template <template <typename TP> class Data, typename T>
void RStarTree<Data, T>::load_root()
{
	if(root==NULL)
	{
#ifndef _USE_HASH_NODE_BUFFER_
		typename map<int, RSTNode<Data, T> *>::iterator iter=node_buf->find(root_page);
#else
		typename unordered_map<int, RSTNode<Data, T> *>::iterator iter=node_buf->find(root_page);
#endif
		if(iter!=node_buf->end())
		{
			//fetch from buffer
			root=iter->second;
		}
		else
		{
			if(root_is_leaf) root=new RSTLeafNode<Data, T>(this, root_page);
			else root=new RSTNonLeafNode<Data, T>(this, root_page);
		}
	}
}

template <template <typename TP> class Data, typename T>
const char *RStarTree<Data, T>::read_header(const char *buffer)
{
	memcpy(&dim, buffer, sizeof(int));
	buffer+=sizeof(int);
	memcpy(&num_data, buffer, sizeof(int));
	buffer+=sizeof(int);
	memcpy(&num_leaf, buffer, sizeof(int));
	buffer+=sizeof(int);
	memcpy(&num_nonleaf, buffer, sizeof(int));
	buffer+=sizeof(int);
	memcpy(&root_is_leaf, buffer, sizeof(bool));
	buffer+=sizeof(bool);
	memcpy(&root_page, buffer, sizeof(int));
	buffer+=sizeof(int);

	return buffer;
}

template <template <typename TP> class Data, typename T>
char *RStarTree<Data, T>::write_header(char *buffer)
{
	memcpy(buffer, &dim, sizeof(int));
	buffer+=sizeof(int);
	memcpy(buffer, &num_data, sizeof(int));
	buffer+=sizeof(int);
	memcpy(buffer, &num_leaf, sizeof(int));
	buffer+=sizeof(int);
	memcpy(buffer, &num_nonleaf, sizeof(int));
	buffer+=sizeof(int);
	memcpy(buffer, &root_is_leaf, sizeof(bool));
	buffer+=sizeof(bool);
	memcpy(buffer, &root_page, sizeof(int));
	buffer+=sizeof(int);

	return buffer;
}

template <template <typename TP> class Data, typename T>
Data<T> *RStarTree<Data, T>::get(int index, Data<T> *data_buf)
{
	load_root();

	return root->get(index, data_buf);
}

template <template <typename TP> class Data, typename T>
void RStarTree<Data, T>::insert(Data<T> *the_data)
{
	//check parameter validity
	if(the_data==NULL) return;

	//we make sure root node is loaded
	load_root();

	//allocate and initialize re-insertion
	reinsrt_lvl=new bool[root->depth+1];
	for(int i=0;i<=root->depth;i++) reinsrt_lvl[i]=false;
	reinsrt_cands->push_front(*the_data);

	//int loop=-1;
	Data<T> *cand_data=new Data<T>(dim);
	RSTNode<Data, T> *new_node;
	while(!reinsrt_cands->empty())
	{
		//get first candidate
		*cand_data=reinsrt_cands->front();
		reinsrt_cands->pop_front();

		//insert recursively to root
		INSERT_RESULT root_insert=root->insert(cand_data, &new_node);

		if(root_insert==INSERT_SPLIT)
		{
			RSTNonLeafNode<Data, T> *new_root=new RSTNonLeafNode<Data, T>(this);
			new_root->child_is_leaf=root_is_leaf;
			new_root->depth=root->depth+1;

			//make root and new_node as two children of new_root
			T *mbr_buf=new T[2*dim];
			root->get_mbr(mbr_buf);
			new_root->enter(mbr_buf, root->page_id, root->get_num_of_data(), root);



			new_node->get_mbr(mbr_buf);
			new_root->enter(mbr_buf, new_node->page_id, new_node->get_num_of_data(), new_node);

			//update root
			root_page=new_root->page_id;
			root=new_root;
			root_is_leaf=false;

			//release
			delete[] mbr_buf;
		}

	}

	//update statistics and flush
	num_data++;
	root->flush();


	//release
	delete cand_data;
	delete[] reinsrt_lvl;
	reinsrt_lvl=NULL;
}

template <template <typename TP> class Data, typename T>
void RStarTree<Data, T>::bulkload_str(Data<T> **the_data_array, int num_data, double load_factor)
{
	//first, we check it the tree is newly created or not
	if(this->num_data!=0||(!root_is_leaf))
	{
		printf("Tree is not empty! Cannot do bulk loading! Program halt!\n");
		exit(-1);
	}

	//step 1, decide how many steps we need to do
	int leaf_load_capacity=(int)(load_factor*this->leaf_capacity);
	int nonleaf_load_capacity=(int)(load_factor*this->nonleaf_capacity);
	//printf("Bulkload leaf capcacity: %d, non-leaf capacity: %d\n", leaf_load_capacity, nonleaf_load_capacity);
	int blk=leaf_load_capacity;

	vector<int> block_size;
	if(num_data>blk)
	{
		block_size.push_back(blk);
		blk*=nonleaf_load_capacity;

		while((num_data>blk)&&((int)block_size.size()<this->dim))
		{
			block_size.push_back(blk);
			blk*=nonleaf_load_capacity;
		}
	}

	//step 2, do sorting in a top-down fashion
	int *data_ids=new int[num_data];
	for(int i=0;i<num_data;i++) data_ids[i]=the_data_array[i]->id;

	int cur_dim=dim-(int)block_size.size();
	std::sort(data_ids, data_ids+num_data, data_compare_center_less<Data, T>(the_data_array, cur_dim));

	int pos=block_size.size()-1;
	int count, blk_start, blk_end;
	cur_dim++;	
	while(pos>0)
	{
		blk=block_size[pos];
		count=num_data/blk;
		blk_start=0;
		blk_end=blk;

		for(int i=0;i<count;i++)
		{
			std::sort(data_ids+blk_start, data_ids+blk_end, data_compare_center_less<Data, T>(the_data_array, cur_dim));
			blk_start+=blk;
			blk_end+=blk;
		}
		if(blk_start<num_data)
			std::sort(data_ids+blk_start, data_ids+num_data, data_compare_center_less<Data, T>(the_data_array, cur_dim));

		cur_dim++;
		pos--;
	}

	//step 3, construct tree in a bottom-up fashion
	deque<RSTNode<Data, T> *> nodes;
	deque<RSTNode<Data, T> *> collected_nodes;

	RSTNode<Data, T> *temp;
	RSTLeafNode<Data, T> *leaf_node=static_cast<RSTLeafNode<Data, T> *>(root);
	RSTNonLeafNode<Data, T> *nonleaf_node=NULL;
	root=NULL;
	root_page=-1;

	//first we construct leaf nodes, now it is a new file and we already have one leaf node as root
	blk=leaf_load_capacity;
	count=num_data/blk;
	blk_start=0;
	blk_end=blk;
	for(int i=0;i<count;i++)
	{
		if(leaf_node==NULL) leaf_node=new RSTLeafNode<Data, T>(this);
		for(int j=blk_start;j<blk_end;j++)
		{
			if(leaf_node->insert(the_data_array[data_ids[j]], &temp)!=INSERT_NONE)
			{
				printf("Fatal error in file: %s, func: %s, line: %d\n", __FILE__, __FUNCTION__, __LINE__);
				printf("Program halt!\n");
				exit(-1);
			}
		}
		leaf_node->nmbr= new T[2 * this->dim];
		leaf_node->get_mbr(leaf_node->nmbr);

		nodes.push_back(leaf_node);
		leaf_node=NULL;

		blk_start+=blk;
		blk_end+=blk;
	}
	if(blk_start<num_data)
	{
		if(leaf_node==NULL) leaf_node=new RSTLeafNode<Data, T>(this);
		for(int j=blk_start;j<num_data;j++)
		{
			if(leaf_node->insert(the_data_array[data_ids[j]], &temp)!=INSERT_NONE)
			{
				printf("Fatal error in file: %s, func: %s, line: %d\n", __FILE__, __FUNCTION__, __LINE__);
				printf("Program halt!\n");
				exit(-1);
			}
		}
		leaf_node->nmbr = new T[2 * this->dim];
		leaf_node->get_mbr(leaf_node->nmbr);
		nodes.push_back(leaf_node);
	}

	//next, we combine the nodes to build up a hierarchy
	blk=nonleaf_load_capacity;
	T *the_mbr=new T[2*this->dim];
	while(nodes.size()>1)
	{
		bool child_leaf=nodes.front()->is_leaf_node();
		int child_level=nodes.front()->depth;

		while(!nodes.empty())
		{
			nonleaf_node=new RSTNonLeafNode<Data, T>(this);
			nonleaf_node->child_is_leaf=child_leaf;
			nonleaf_node->depth=child_level;
			if((int)nodes.size()>=blk)
			{
				for(int i=0;i<blk;i++)
				{
					RSTNode<Data, T> *temp=nodes.front();
					nodes.pop_front();

					nonleaf_node->enter(temp->get_mbr(the_mbr), temp->page_id, temp->get_num_of_data(), temp);
				}
			}
			else
			{
				while(!nodes.empty())
				{
					RSTNode<Data, T> *temp=nodes.front();
					nodes.pop_front();

					nonleaf_node->enter(temp->get_mbr(the_mbr), temp->page_id, temp->get_num_of_data(), temp);
				}
			}

			collected_nodes.push_back(nonleaf_node);
		}

		nodes.swap(collected_nodes);
	}

	//finally, we make root correct
	root=nodes.front();
	root_is_leaf=root->is_leaf_node();
	root_page=root->page_id;

	this->num_data=num_data;

	delete[] data_ids;
	delete[] the_mbr;
}

#endif

}