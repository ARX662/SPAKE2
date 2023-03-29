#ifndef COMPLEX
#define COMPLEX

#include "uint256_t.h"
#define bigint uint256_t

#include "Fp.h"

class Complex : public bigint {
	public:
		Fp field;
		uint256_t real, imag;

		Complex();
		Complex(Fp field, bigint real);
		Complex(Fp field, bigint real, bigint imag);

		Complex conjugate();

		Complex multiply(Complex p);
		Complex square(Complex p);
		Complex divide(Complex p);

		string str();
};

#endif
