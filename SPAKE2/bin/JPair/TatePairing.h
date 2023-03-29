#ifndef TATEPAIRING
#define TATEPAIRING

#include "Complex.h"
#include "Random.h"

// Tate pairing implementation.
class TatePairing {
	public:
	   EllipticCurve ec, twisted;
	   Fp field, gt;
	   bigint order, finalExponent, cof;
	
	   TatePairing();
	   TatePairing(EllipticCurve curve, bigint groupOrder, bigint coFactor);
	
	   EllipticCurve twist(EllipticCurve e);

	   Point RandomPointInG1(Random rd);
	   //Point RandomPointInG1_2();	

	   Complex compute(Point P, Point Q);

	   void naf(uint8_t *r, bigint k, uint8_t w);

	   Complex encDbl(JacobPoint &P, Point Q);
	   Complex encAdd(JacobPoint &A, Point P, Point Q);
};

#endif
