#ifndef _PAGE_FILE_H_
#define _PAGE_FILE_H_

#include <cstring>
#include <list>
#include <vector>
#include <fstream>
using namespace std;

#include "GenericTool.h"

//Provide functionality for cached page file
//we use one page to put header for maintennance
//and the header of objects using this file as storage
namespace dblsh {

class PageFile
{
public:
	fstream *f; // lower level file object
	char *fname; // stored filed name
	bool new_file; // is this a new file?

	int page_len; // the page length
	int page_num; // total number of pages

	//cache
	enum c_policy {C_FULLMEM=0, C_LRU=1, C_MRU=2, C_NO_CACHE=3};

	c_policy policy; // current policy
	int cache_size; // the size of cache
	char *buf_page; // one page buffer used to store page when no cache is used
	int num_cached; // the number of cached pages
	vector<char *> *cache; // the cache map from page id to buffer pointer
	vector<bool> *c_dirty; // the cache map from page id to if it is dirty (been modified)
	list<int> *use_list; // stores the page id list fron (most recent) -> end (least recent)
	vector<list<int>::iterator> *list_pos; // the cache map from page id to buffer pointer

	//statistics
	int page_access;

	// p_len is effective only for new files
	// pol is cache policy of {C_FULLMEM, C_LRU, C_MRU, C_NO_CACHE}
	// c_size is size of cache
	PageFile(const char *name, int p_len, const char *path=NULL, c_policy pol=C_FULLMEM, int c_size=0, bool force_new=false);
	~PageFile();

	void Construct(const char *name, int p_len, const char *path, c_policy pol, int c_size, bool force_new);
	void EmptyValue();
	void CleanValue();

	const char *read_header(); // read header block without blockfile internal header
	void set_header(const char *header); // set block fileheader without blockfile internal header
	int get_header_size() { return page_len-2*sizeof(int); } // get actual header length

	//the index is the number except header
	//i.e, header is -1, and the actual page after header is 0
	const char *read_page(int index); // reads page
	bool write_page(const char *p, int index); // write page
	int append_page(const char *p); // returns page number
	bool truncate_pages(int num); // deletes the last num pages

	//write all page back to disk
	void flush();

};

#endif

PageFile::PageFile(const char *name, int p_len, const char *path, c_policy pol, int c_size, bool force_new)
{
	EmptyValue();

	Construct(name, p_len, path, pol, c_size, force_new);
}

PageFile::~PageFile()
{
	CleanValue();
}

void PageFile::Construct(const char *name, int p_len, const char *path, c_policy pol, int c_size, bool force_new)
{
	//assign values
	int fn_len=GenericTool::GetCombinedPath(path, name, NULL);
	fname=new char[fn_len];
	GenericTool::GetCombinedPath(path, name, fname);
	page_len=p_len;
	page_num=1; //we always have a header page
	policy=pol;
	if((pol==C_FULLMEM)||(pol==C_NO_CACHE)) cache_size=0;
	else cache_size=c_size;

	//test if we can find the file
	if(GenericTool::JudgeExistence(fname, force_new))
	{
		//file exists, then we read parameter from file
		f=new fstream(fname, ios::in | ios::out | ios::binary);
		new_file=false;

		if(!f)
		{
			printf("Open File %s Failed!\n", fname);
			exit(-1);
		}

		f->read((char *)&page_len, sizeof(int));
		f->read((char *)&page_num, sizeof(int));

		//allocate one buffer page to use
		buf_page=new char[page_len];
	}
	else
	{
		//file does not exist, we first create a zero-sized file by specify ios::out
		//then we close it and re-open it with ios::in and ios::out
		//because directly go on with only ios::out makes us cannot read
		f=new fstream(fname, ios::in | ios::out | ios::binary);
		new_file=true;

		if(!f->is_open())
		{
			printf("Open File %s Failed!\n", fname);
			exit(-1);
		}

		//we then allocate one buffer page to use, and write data to file
		buf_page=new char[page_len];
		memset(buf_page, 0, page_len);
		memcpy(buf_page, &page_len, sizeof(int));
		memcpy(buf_page+sizeof(int), &page_num, sizeof(int));
		f->write(buf_page, page_len);
	}

	char *buf=NULL;

	if(f->fail()) printf("fail start!");

	switch(policy)
	{
	case C_FULLMEM:
		cache=new vector<char *>();
		cache->resize(page_num, NULL);
		c_dirty=new vector<bool>();
		c_dirty->resize(page_num, false);

		//make everything cached
		f->seekg(0, ios::beg);
		for(int i=0;i<page_num;i++)
		{
			buf=new char[page_len];
			f->read(buf, page_len);
			cache->at(i)=buf;
			c_dirty->at(i)=false;
		}

		break;
	case C_LRU:
	case C_MRU:
		cache=new vector<char *>();
		cache->resize(page_num, NULL);
		c_dirty=new vector<bool>();
		c_dirty->resize(page_num, false);
		use_list=new list<int>();
		list_pos=new vector<list<int>::iterator>();
		list_pos->resize(page_num, use_list->end());

		//make header cached
		buf=new char[page_len];
		f->seekg(0, ios::beg);
		f->read(buf, page_len);
		cache->at(0)=buf;
		c_dirty->at(0)=false;
		use_list->push_front(0);
		list_pos->at(0)=use_list->begin();

		cache_size=max(cache_size, 1);
		num_cached=1;

		break;
	}
}

void PageFile::EmptyValue()
{
	f=NULL;
	fname=NULL;
	new_file=false;
	page_len=0;
	page_num=0;
	policy=C_FULLMEM;
	cache_size=0;
	buf_page=NULL;
	num_cached=0;
	cache=NULL;
	c_dirty=NULL;
	use_list=NULL;
	list_pos=NULL;
	page_access=0;
}

void PageFile::CleanValue()
{
	//flush();

	if(f!=NULL) delete f;
	if(fname!=NULL) delete[] fname;
	if(buf_page!=NULL) delete[] buf_page;
	if(cache!=NULL)
	{
		for(vector<char *>::iterator iter=cache->begin();iter!=cache->end();++iter)
		{
			delete[] (*iter);
		}
		delete cache;
	}
	if(c_dirty!=NULL) delete c_dirty;
	if(use_list!=NULL) delete use_list;
	if(list_pos!=NULL) delete list_pos;

	EmptyValue();
}

const char *PageFile::read_header()
{
	//return the address of page 0 plus the size of page_len+page_num
	return read_page(-1)+2*sizeof(int);
}

void PageFile::set_header(const char *header)
{
	memcpy(buf_page, &page_len, sizeof(int));
	memcpy(buf_page+sizeof(int), &page_num, sizeof(int));
	memcpy(buf_page+2*sizeof(int), header, page_len-2*sizeof(int));
	write_page(buf_page, -1);
}

const char *PageFile::read_page(int index)
{
	index++; //transfer to correct page index
	if((index<0)||(index>=page_num)) return NULL;

	//statistics
	page_access++;

	char *buf=NULL;
	int evict_ind=-1;

	switch(policy)
	{
	case C_FULLMEM:
		return cache->at(index);
		break;
	case C_LRU:
		if(cache->at(index)==NULL)
		{
			//no cache hit!
			if(num_cached<cache_size)
			{
				//cache underfull
				buf=new char[page_len];
				num_cached++;
			}
			else
			{
				//cache full, evict one page
				evict_ind=use_list->back();
				buf=cache->at(evict_ind);
				if(c_dirty->at(evict_ind))
				{
					//if dirty then we nned to write back to disk
					f->seekp(evict_ind*page_len, ios::beg);
					f->write(buf, page_len);
					c_dirty->at(evict_ind)=false;
				}
				cache->at(evict_ind)=NULL;
				use_list->pop_back();
				list_pos->at(evict_ind)=use_list->end();
			}

			//we cache new page
			f->seekg(index*page_len, ios::beg);
			f->read(buf, page_len);
			cache->at(index)=buf;
			c_dirty->at(index)=false;
			use_list->push_front(index);
			list_pos->at(index)=use_list->begin();
		}
		else
		{
			buf=cache->at(index);
			use_list->erase(list_pos->at(index));
			use_list->push_front(index);
			list_pos->at(index)=use_list->begin();
		}

		return buf;
		break;
	case C_MRU:
		if(cache->at(index)==NULL)
		{
			//no cache hit!
			if(num_cached<cache_size)
			{
				//cache underfull
				buf=new char[page_len];
				num_cached++;
			}
			else
			{
				//cache full, evict one page
				evict_ind=use_list->front();
				buf=cache->at(evict_ind);
				if(c_dirty->at(evict_ind))
				{
					//if dirty then we nned to write back to disk
					f->seekp(evict_ind*page_len, ios::beg);
					f->write(buf, page_len);
					c_dirty->at(evict_ind)=false;
				}
				cache->at(evict_ind)=NULL;
				use_list->pop_front();
				list_pos->at(evict_ind)=use_list->end();
			}

			//we cache new page
			f->seekg(index*page_len, ios::beg);
			f->read(buf, page_len);
			cache->at(index)=buf;
			c_dirty->at(index)=false;
			use_list->push_front(index);
			list_pos->at(index)=use_list->begin();
		}
		else
		{
			buf=cache->at(index);
			use_list->erase(list_pos->at(index));
			use_list->push_front(index);
			list_pos->at(index)=use_list->begin();
		}

		return buf;
		break;
	case C_NO_CACHE:
		f->seekg(index*page_len, ios::beg);
		f->read(buf_page, page_len);
		return buf_page;
		break;
	}

	return NULL;
}

bool PageFile::write_page(const char *p, int index)
{
	index++; //transfer to correct page index
	if((index<0)||(index>=page_num)) return false;

	//statistics
	page_access++;

	char *buf=NULL;
	int evict_ind=-1;

	switch(policy)
	{
	case C_FULLMEM:
		memcpy(cache->at(index), p, page_len);
		c_dirty->at(index)=true;
		break;
	case C_LRU:
		if(cache->at(index)==NULL)
		{
			//no cache hit!
			if(num_cached<cache_size)
			{
				//cache underfull
				buf=new char[page_len];
				num_cached++;
			}
			else
			{
				//cache full, evict one page
				evict_ind=use_list->back();
				buf=cache->at(evict_ind);
				if(c_dirty->at(evict_ind))
				{
					//if dirty then we need to write back to disk
					f->seekp(evict_ind*page_len, ios::beg);
					f->write(buf, page_len);
					c_dirty->at(evict_ind)=false;
				}
				cache->at(evict_ind)=NULL;
				use_list->pop_back();
				list_pos->at(evict_ind)=use_list->end();
			}

			//we cache new page
			memcpy(buf, p, page_len);
			cache->at(index)=buf;
			c_dirty->at(index)=true;
			use_list->push_front(index);
			list_pos->at(index)=use_list->begin();
		}
		else
		{
			memcpy(cache->at(index), p, page_len);
			c_dirty->at(index)=true;
			use_list->erase(list_pos->at(index));
			use_list->push_front(index);
			list_pos->at(index)=use_list->begin();
		}
		break;
	case C_MRU:
		if(cache->at(index)==NULL)
		{
			//no cache hit!
			if(num_cached<cache_size)
			{
				//cache underfull
				buf=new char[page_len];
				num_cached++;
			}
			else
			{
				//cache full, evict one page
				evict_ind=use_list->front();
				buf=cache->at(evict_ind);
				if(c_dirty->at(evict_ind))
				{
					//if dirty then we nned to write back to disk
					f->seekp(evict_ind*page_len, ios::beg);
					f->write(buf, page_len);
					c_dirty->at(evict_ind)=false;
				}
				cache->at(evict_ind)=NULL;
				use_list->pop_front();
				list_pos->at(evict_ind)=use_list->end();
			}

			//we cache new page
			memcpy(buf, p, page_len);
			cache->at(index)=buf;
			c_dirty->at(index)=true;
			use_list->push_front(index);
			list_pos->at(index)=use_list->begin();
		}
		else
		{
			memcpy(cache->at(index), p, page_len);
			c_dirty->at(index)=true;
			use_list->erase(list_pos->at(index));
			use_list->push_front(index);
			list_pos->at(index)=use_list->begin();
		}
		break;
	case C_NO_CACHE:
		f->seekp(index*page_len, ios::beg);
		f->write(p, page_len);
		break;
	}

	return true;
}

int PageFile::append_page(const char *p)
{
	int index=page_num++;

	//statistics
	page_access++;

	char *buf=NULL;
	int evict_ind=-1;

	switch(policy)
	{
	case C_FULLMEM:
		buf=new char[page_len];
		memcpy(buf, p, page_len);
		cache->push_back(buf);
		c_dirty->push_back(true);
		break;
	case C_LRU:
		if(num_cached<cache_size)
		{
			//cache underfull
			buf=new char[page_len];
			num_cached++;
		}
		else
		{
			//cache full, evict one page
			evict_ind=use_list->back();
			buf=cache->at(evict_ind);
			if(c_dirty->at(evict_ind))
			{
				//if dirty then we nned to write back to disk
				f->seekp(evict_ind*page_len, ios::beg);
				f->write(buf, page_len);
				c_dirty->at(evict_ind)=false;
			}
			cache->at(evict_ind)=NULL;
			use_list->pop_back();
			list_pos->at(evict_ind)=use_list->end();
		}

		//we cache new page
		memcpy(buf, p, page_len);
		cache->push_back(buf);
		c_dirty->push_back(true);
		use_list->push_front(index);
		list_pos->push_back(use_list->begin());
		break;
	case C_MRU:
		if(num_cached<cache_size)
		{
			//cache underfull
			buf=new char[page_len];
			num_cached++;
		}
		else
		{
			//cache full, evict one page
			evict_ind=use_list->front();
			buf=cache->at(evict_ind);
			if(c_dirty->at(evict_ind))
			{
				//if dirty then we nned to write back to disk
				f->seekp(evict_ind*page_len, ios::beg);
				f->write(buf, page_len);
				c_dirty->at(evict_ind)=false;
			}
			cache->at(evict_ind)=NULL;
			use_list->pop_front();
			list_pos->at(evict_ind)=use_list->end();
		}

		//we cache new page
		memcpy(buf, p, page_len);
		cache->push_back(buf);
		c_dirty->push_back(true);
		use_list->push_front(index);
		list_pos->push_back(use_list->begin());
		break;
	case C_NO_CACHE:
		f->seekp(index*page_len, ios::beg);
		f->write(p, page_len);
		break;
	}

	return index-1;
}

//this is a cross-platform implementations
//since this function is slow, so avoid 
bool PageFile::truncate_pages(int num)
{
	//make sure all things are written back to disk
	flush();

	//release all memory cache
	if(cache!=NULL)
	{
		for(vector<char *>::iterator iter=cache->begin();iter!=cache->end();++iter)
		{
			delete[] (*iter);
		}
		delete cache;
	}
	if(c_dirty!=NULL) delete c_dirty;
	if(use_list!=NULL) delete use_list;
	if(list_pos!=NULL) delete list_pos;

	//reset values
	num_cached=0;
	page_num=max(page_num-num, 1);

	//open temporary file
	fstream *ftemp=new fstream("temp_page_file", ios::out | ios::binary);

	//copy file in pages
	f->seekg(0, ios::beg);
	ftemp->seekp(0, ios::beg);
	for(int i=0;i<page_num;i++)
	{
		f->read(buf_page, page_len);
		ftemp->write(buf_page, page_len);
	}

	//close both file
	f->close();
	delete f;
	ftemp->close();
	delete ftemp;

	//rename file
	if(remove(fname)||rename("temp_page_file", fname))
	{
		printf("Truncate file fail! Page file object is corrupted!\n");
		return false;
	}

	//reconstruct data structure
	char *buf=NULL;

	f=new fstream(fname, ios::in | ios::out | ios::binary);
	switch(policy)
	{
	case C_FULLMEM:
		cache=new vector<char *>();
		cache->resize(page_num, NULL);
		c_dirty=new vector<bool>();
		c_dirty->resize(page_num, false);

		//make everything cached
		f->seekg(0, ios::beg);
		for(int i=0;i<page_num;i++)
		{
			buf=new char[page_len];
			f->read(buf, page_len);
			cache->at(i)=buf;
			c_dirty->at(i)=false;
		}

		break;
	case C_LRU:
	case C_MRU:
		cache=new vector<char *>();
		cache->resize(page_num, NULL);
		c_dirty=new vector<bool>();
		c_dirty->resize(page_num, false);
		use_list=new list<int>();
		list_pos=new vector<list<int>::iterator>();
		list_pos->resize(page_num, use_list->end());

		//make header cached
		buf=new char[page_len];
		f->seekg(0, ios::beg);
		f->read(buf, page_len);
		cache->at(0)=buf;
		c_dirty->at(0)=false;
		use_list->push_front(0);
		list_pos->at(0)=use_list->begin();

		num_cached=1;

		break;
	}

	return true;
}

void PageFile::flush()
{
	switch(policy)
	{
	case C_FULLMEM:
		//check if we can flush
		if((c_dirty==NULL)||(cache==NULL)) return;

		for(int i=0;i<page_num;i++)
		{
			if(c_dirty->at(i))
			{
				c_dirty->at(i)=false;
				f->seekp(i*page_len, ios::beg);
				f->write(cache->at(i), page_len);
			}
		}
		break;
	case C_LRU:
	case C_MRU:
		//check if we can flush
		if((use_list==NULL)||(c_dirty==NULL)||(cache==NULL)) return;

		for(list<int>::iterator iter=use_list->begin();iter!=use_list->end();++iter)
		{
			int ind=*iter;
			if(c_dirty->at(ind))
			{
				c_dirty->at(ind)=false;
				f->seekp(ind*page_len, ios::beg);
				f->write(cache->at(ind), page_len);
			}
		}
		break;
	case C_NO_CACHE:
		break;
	}

	//make page_len and page_num synchronized
	f->seekp(0, ios::beg);
	f->write((char *)&page_len, sizeof(int));
	f->write((char *)&page_num, sizeof(int));
	f->flush();
}

}