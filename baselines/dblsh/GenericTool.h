//This file contains some neat implementations of useful tool functions
//by GS
#ifndef _GENERIC_TOOL_H_
#define _GENERIC_TOOL_H_

#pragma once

//#include "Common.h"

//Cross Platform snprintf
#include <cstdio>
#include <cstdarg>

#ifdef _MSC_VER
//Under vc, we have to use some simulation
int msvc_snprintf(char *str, size_t size, const char *format, ...);
#define c99_snprintf msvc_snprintf
#else
#ifdef __GNUC__
//Under g++, we just directly use snprintf
#define c99_snprintf snprintf
#else
//For other compiler, we output error
int other_snprintf(char *str, size_t size, const char *format, ...);
#define c99_snprintf other_snprintf
#endif
#endif

//Random Number Handling using MT19937 library
#include "mt19937ar.h"

//init seed function
#define setseed(seed) init_genrand(seed)

//can change to different variaion in mt19937 library
//this version get double value in [0,1)
#define getrand() genrand_real2()

//Some Cross Platform Important Functions
#include <cmath>
#include <ctime>
#include <algorithm>
#include <iterator>
#include <cstdio>
#include <cerrno>
#include <cfloat>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>

namespace dblsh {

class GenericTool
{
public:
	//generic purpose tool function
	static int CountBit(int num);

	//for file manipulation
	static bool CheckPathExistence(const char *path);
	static int RegularizeDirPath(const char *path, char *buffer);
	static void EnsurePathExistence(const char *path);
	static int GetCombinedPath(const char *dir, const char *file, char *buffer);
	static bool JudgeExistence(const char *full_path, bool force_new);
	static int ChangeFileExtension(const char *full_path, const char *new_ext, char *buffer);

	//random number related and data generation
	static double GetGaussianRandom(double mean, double sigma);
	
	//some useful templates
	template <typename T> static T DotProduct(int dim, T *a, T *b);
	template <typename T> static T GetSign(T val);

	//for simple matrix operation
	//T should be float, double or long double to make sense
	template <typename T> static T **AllocateMatrix(int m, int n); //we also assign every element with zero
	template <typename T> static T **CopyMatrix(T **mat, int m, int n);
	template <typename T> static void ReleaseMatrix(T **mat, int m, int n);
	template <typename T> static void OutMatrix(T **mat, int m, int n);
	template <typename T> static bool GaussJordanElimination(T **mat, int m, int n); //Gaussian Elimination for matrix m(total row)x n(total column)
	template <typename T> static bool InverseMatrix(T **mat, int m, T **inv); //inverse mxm matrix

	//functions for discretization
	//typename T should be floating type
	template <typename T> static int DiscreteValueFloor(T val, int seg_num);
	template <typename T> static int DiscreteValueFloor(T val, int seg_num, T val_min, T val_max);
	template <typename T> static int DiscreteValueCeil(T val, int seg_num);
	template <typename T> static int DiscreteValueCeil(T val, int seg_num, T val_min, T val_max);
	template <typename T> static T ContinuousValueFloor(int seg_id, int seg_num);
	template <typename T> static T ContinuousValueFloor(int seg_id, int seg_num, T val_min, T val_max);
	template <typename T> static T ContinuousValueCeil(int seg_id, int seg_num);
	template <typename T> static T ContinuousValueCeil(int seg_id, int seg_num, T val_min, T val_max);

	//for indirect compare
	template <typename T>
	struct indirect_comp_less
	{
		T *ref_data;

		indirect_comp_less(T *scores) : ref_data(scores) {}
		bool operator()(const int id1, const int id2) const
		{
			if(ref_data[id1]<ref_data[id2]) return true;
			else return false;
		}
	};

	template <typename T>
	struct indirect_comp_greater
	{
		T *ref_data;

		indirect_comp_greater(T *scores) : ref_data(scores) {}
		bool operator()(const int id1, const int id2) const
		{
			if(ref_data[id1]>ref_data[id2]) return true;
			else return false;
		}
	};
};

inline int GenericTool::CountBit(int num)
{
	int count=0;
	while(num)
	{
		count++;
		num&=(num-1); //every time we reduce the number of "1" in the binary representation of num by 1
	}
	return count;
}

template<typename T>
inline T GenericTool::DotProduct(int dim, T *a, T *b)
{
	T res=0;
	for(int i=0;i<dim;i++) res+=a[i]*b[i];
	return res;
}

template <typename T>
inline T GenericTool::GetSign(T val)
{
	return (T)((val>0)-(val<0));
}

//templates for matri operations
template <typename T>
inline T **GenericTool::AllocateMatrix(int m, int n)
{
	T **mat=new (T*[m]);
	for(int i=0;i<m;i++)
	{
		mat[i]=new T[n];
		memset(mat[i], 0, n*sizeof(T));
	}
	return mat;
}

template <typename T>
inline T **GenericTool::CopyMatrix(T **mat, int m, int n)
{
	T **copy_mat=AllocateMatrix<T>(m, n);
	for(int i=0;i<m;i++) memcpy(copy_mat[i], mat[i], n*sizeof(T));

	return copy_mat;
}

template <typename T>
inline void GenericTool::ReleaseMatrix(T **mat, int m, int n)
{
	for(int i=0;i<m;i++) delete[] mat[i];
	delete[] mat;
}

template <typename T>
inline void GenericTool::OutMatrix(T **mat, int m, int n)
{
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<n;j++) cout <<mat[i][j]<<' ';
		cout <<endl;
	}
}

template <typename T>
inline bool GenericTool::GaussJordanElimination(T **mat, int m, int n)
{
	int i=0;
	int j=0;

	while((i<m)&&(j<n))
	{
		//find pivot in column j, starting fron row i
		int maxi=i;
		T mx=mat[i][j];
		for(int k=i+1;k<m;k++)
		{
			if(abs(mat[k][j])>abs(mx))
			{
				mx=mat[k][j];
				maxi=k;
			}
		}

		//if max is zero then we cannot continue
		if(mx!=0)
		{
			//swap row
			if(maxi!=i)
			{
				T *temp_row=mat[i];
				mat[i]=mat[maxi];
				mat[maxi]=temp_row;
			}

			for(int k=j;k<n;k++) mat[i][k]/=mx;
			for(int k=0;k<m;k++)
			{
				if(k==i) continue;

				T mul=mat[k][j];
				for(int l=j;l<n;l++) mat[k][l]-=mul*mat[i][l];
			}
		}
		else return false;

		i++;
		j++;
	}

	return true;
}

template <typename T>
inline bool GenericTool::InverseMatrix(T **mat, int m, T **inv)
{
	T **temp=AllocateMatrix<T>(m, 2*m);
	for(int i=0;i<m;i++)
	{
		memcpy(temp[i], mat[i], m*sizeof(T));
		temp[i][m+i]=1;
	}

	if(!GaussJordanElimination(temp, m, 2*m))
	{
		ReleaseMatrix(temp, m, 2*m);
		return false;
	}
	else
	{
		for(int i=0;i<m;i++) memcpy(inv[i], temp[i]+m, m*sizeof(T));
		ReleaseMatrix(temp, m, 2*m);
		return true;
	}
}

#endif

//Cross Platform snprintf
#ifdef _MSC_VER
//Under vc, we have to use some simulation
int msvc_snprintf(char *str, size_t size, const char *format, ...)
{
	int count;

	va_list ap;
	va_start(ap, format);
	count=_vscprintf(format, ap);
	_vsnprintf_s(str, size, _TRUNCATE, format, ap);
	va_end(ap);

	return count;
}
#else
#ifdef __GNUC__
//We have already use the snprintf directly
#else
//For other compiler, we output error
int other_snprintf(char *str, size_t size, const char *format, ...)
{
	printf("Not Implemented!\n");

	return -1;
}
#endif
#endif

//file related tool functions
bool GenericTool::CheckPathExistence(const char *path)
{
	errno=0;
	return (!rename(path, path) || (errno!=2)); //can rename or errno is not "file not exist"
}

//if buffer is NULL, thren return the number of buffer required including '\0'
int GenericTool::RegularizeDirPath(const char *path, char *buffer)
{
	//default path assignment
#ifdef WIN32
	if(path==NULL) path="../"; //win32 default
#else
	if(path==NULL) path="./"; //linux default
#endif	

	int len=strlen(path);

	if(len==0) //path=""
	{
		if(buffer!=NULL) buffer[0]='\0';
		return 1;
	}

	//only check last character is '\\' or '/'
	if((path[len-1]=='\\')||(path[len-1]=='/'))
	{
		//OK, just copy
		if(buffer!=NULL)
		{
			memcpy(buffer, path, (len+1)*sizeof(char));
			
			//since UNIX does not recognize '\\' in path but windows do recognize '/', we replace all '\\' to '/'
			for(int i=0;i<len;i++) if(buffer[i]=='\\') buffer[i]='/';
		}			
	}
	else
	{
		//not OK, need one more
		len++;
		if(buffer!=NULL)
		{
			c99_snprintf(buffer, len+1, "%s/", path); //use linux default '/', generally windows can recognize

			//since UNIX does not recognize '\\' in path but windows do recognize '/', we replace all '\\' to '/'
			for(int i=0;i<len;i++) if(buffer[i]=='\\') buffer[i]='/';
		}
	}
	
	return len+1;
}

void GenericTool::EnsurePathExistence(const char *path)
{
	if((path!=NULL)&&!CheckPathExistence(path))
	{
		int len=strlen(path);
		char *dir_path=new char[len+10];
		int ret=c99_snprintf(dir_path, len+10, "mkdir %s -p", path); // unix
		system(dir_path); //use mkdir, which is suported by a variety of OS
		delete[] dir_path;
	}
}

//if buffer is NULL, thren return the number of buffer required including '\0'
int GenericTool::GetCombinedPath(const char *dir, const char *file, char *buffer)
{
	int len_dir=RegularizeDirPath(dir, buffer);
	int len_file=strlen(file);
	int len=len_dir+len_file;

	if(buffer!=NULL) memcpy(buffer+len_dir-1, file, (len_file+1)*sizeof(char));

	return len;
}

//if we are forced to create new file, then we will remove orginal file and return false
bool GenericTool::JudgeExistence(const char *full_path, bool force_new)
{
	bool new_buffer=false;
	int path_len=strlen(full_path);

	char *buffer=new char[path_len+1];
	memcpy(buffer, full_path, (path_len+1)*sizeof(char));
	for(int i=0;i<path_len;i++) if(buffer[i]=='\\') buffer[i]='/';

	if(CheckPathExistence(buffer)&&!force_new)
	{
		delete[] buffer;
		return true;
	}
	else
	{
		errno=0;
		if(remove(buffer)&&errno!=2) //if cannot remove and the reason is not that file not exist
		{
			printf("Cannot remove file: %s for a forced new file creation.\n", buffer);
			exit(-1);
		}
		if(force_new)
		{
			int ind;
			for(ind=path_len-1;ind>=0;ind--) if(buffer[ind]=='/') break;
			ind++;
			char temp=buffer[ind];
			buffer[ind]='\0';
			EnsurePathExistence(buffer);
			buffer[ind]=temp;
			fstream f(buffer, ios::out | ios::trunc); //make sure file exists for later open with zero size
			f.close();
		}
		delete[] buffer;
		return false;
	}
}

int GenericTool::ChangeFileExtension(const char *full_path, const char *new_ext, char *buffer)
{
	int o_len=strlen(full_path);
	int ext_len=strlen(new_ext);

	int pos=o_len-1;
	while((pos>=0)&&(full_path[pos]!='\\')&&(full_path[pos]!='/')&&(full_path[pos]!='.')) pos--;
	if(full_path[pos]!='.')
	{
		if(buffer!=NULL)
		{
			memcpy(buffer, full_path, o_len*sizeof(char));
			buffer[o_len]='.';
		}
		pos=o_len+1;
	}
	else
	{
		pos++;
		if(buffer!=NULL) memcpy(buffer, full_path, pos*sizeof(char));
	}

	if(buffer!=NULL) memcpy(buffer+pos, new_ext, (ext_len+1)*sizeof(char));

	return pos+ext_len+1;
}

//Some Cross Platform Important Functions
double GenericTool::GetGaussianRandom(double mean, double sigma)
{
	double v1, v2, s;

	do
	{
		v1=2*getrand()-1;
		v2=2*getrand()-1;
		s=v1*v1+v2*v2;
	} while(s>=1.0);

	if(s==0.0) return mean;
	else return v1*sqrt(-2.0*log(s)/s)*sigma+mean;
}

}