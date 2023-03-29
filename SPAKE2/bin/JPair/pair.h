#ifndef PAIR_2
#define PAIR_2

template<class K, class V>
struct pair_2
{
	K first;
	V second;
	__host__ __device__ pair_2(K _f, V _s)
	: first(_f), second(_s)
	{}
};

#endif

