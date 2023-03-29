#ifndef FP
#define FP

#include "Random.h"

// Defines finite fields of characteristics p, where p is a prime number.
// All the arithmetic operations are done modulo p.

class Complex;

class Fp {
	public:
	   bigint mod;

	   Fp();
	   Fp(bigint mod);

	   bigint add(bigint a, bigint b);
	   bigint subtract(bigint a, bigint b);
	   bigint multiply(bigint a, bigint b);
	   bigint divide(bigint a, bigint b);

	   bigint square(bigint n);
	   bigint squareRoot(bigint val);	
	   bigint power(bigint x, bigint y);
	   bigint inverse(bigint a);
	   bigint negate(bigint a);

	   Complex square(Complex p);
	   Complex multiply(Complex a, Complex b);
	   Complex power(Complex &u, bigint k);

	   int window(bigint x, int i, int* nbs, int* nzs, int w);
	   bool getBitByDegree(int deg, int8_t* ba, int bits);

	   JacobPoint aToJ(Point p);
	   Point jToA(JacobPoint p);

	   bigint randomElement(Random rd);
	   bigint* LucasSequence(bigint p, bigint q, bigint k, bigint re[]);
};

#endif
