#ifndef ELLIPTICCURVE
#define ELLIPTICCURVE

#include "Random.h"

// Defines elliptic curves over GFP
// The Weierstrass equation of the curves is: Y^2+a_1XY+a_3Y=X^3+a_2X^2+a_4X+a_6 or Y^2=X^3+aX+b.
class EllipticCurve {
	public:
	   Fp fp;
	   bigint a4, a6;
	   bool opt;

	   EllipticCurve();
	   EllipticCurve(Fp fp, bigint a, bigint b);

	   Point add(Point p1, Point p2);
	   Point subtract(Point p1, Point p2);

	   Point negate(Point p1);

	   Point multiply(Point p, bigint k);

	   JacobPoint jMultiplyMut(Point p, bigint k);

	   void jDblMut(JacobPoint *p);

	   void jAddMut(JacobPoint *p, Point *q);

	   Point randomPoint(Random rd);
};

#endif
