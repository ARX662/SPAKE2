#ifndef RANDOM
#define RANDOM

// Random Number Generator
class Random {
	public:
	   unsigned int seed;
	   Random();
	   Random(unsigned int s);
	   unsigned int nextInt();
};

#endif
