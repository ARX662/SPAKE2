#include "Random.h"
#include "pair.h"
#include "point.h"
#include "Fp.h"
#include "EllipticCurve.h"

EllipticCurve::EllipticCurve() {}

EllipticCurve::EllipticCurve(Fp fp, bigint a, bigint b) {
	this->fp = fp;
	this->a4 = a;
	this->a6 = b;
	
	if (a4 == this->fp.negate(bigint(3)))
		this->opt = true;
	else 
		this->opt = false;
}

Point EllipticCurve::add(Point p1, Point p2) {
	bigint top = fp.subtract(p2.y, p1.y);
	bigint bottom = fp.subtract(p2.x, p1.x);
	bigint lambda = fp.multiply(top, fp.inverse(bottom));

	bigint x3 = fp.multiply(lambda, lambda);
	x3 = fp.subtract(x3, p1.x);
	x3 = fp.subtract(x3, p2.x);

	bigint y3 = fp.multiply(lambda, p1.x);
	y3 = fp.subtract(y3, p1.y);
	y3 = fp.subtract(y3, fp.multiply(lambda, x3));

	return Point(x3, y3);
}

Point EllipticCurve::subtract(Point p1, Point p2) {
	return add(p1, negate(p2));
}

Point EllipticCurve::negate(Point p1) {
	bigint y2 = fp.negate(p1.y);
	return Point(p1.x, y2);
}

Point EllipticCurve::multiply(Point p, bigint k) {
	return fp.jToA( jMultiplyMut(p,k) );
}


JacobPoint EllipticCurve::jMultiplyMut(Point p, bigint k) {
	if (k.sign) {
		p = negate(p);
	}

	int degree = k.bits() - 2;

	JacobPoint result= fp.aToJ(p);

	bigint p2degree(1);
	p2degree <<= degree;

	for(int i=degree; i>=0; i--) {
		jDblMut(&result);
		if((k & p2degree) != bigint(0))
			jAddMut(&result, &p);
		p2degree>>=1;
	}
	return result;
}

void EllipticCurve::jDblMut(JacobPoint *p) {
	bigint t1 = fp.square(p->y);
	
	bigint t2 = fp.multiply(p->x, t1);
	t2 = fp.add(t2, t2);
	t2 = fp.add(t2, t2);

	bigint t3 = fp.square(t1);
	t3 = fp.add(t3, t3);
	t3 = fp.add(t3, t3);
	t3 = fp.add(t3, t3);

	bigint t4 = fp.square(p->z);

	bigint t5;
	if (opt) {
		t5 = fp.multiply(fp.subtract(p->x,t4), fp.add(p->x,t4));
		t5 = fp.add(t5, fp.add(t5, t5));
	} else {
		t5 = fp.square(p->x);
		t5 = fp.add(t5, fp.add(t5, t5));
		t5 = fp.add(t5, fp.multiply(this->a4, fp.square(t4)));
	}

	bigint x3 = fp.square(t5);
	x3 = fp.subtract(x3, fp.add(t2, t2));

	bigint y3 = fp.multiply(t5, fp.subtract(t2, x3));
	y3 = fp.subtract(y3, t3);

	bigint z3 = fp.multiply(p->y, p->z);
	z3 = fp.add(z3, z3);

	p->x = x3;
	p->y = y3;
	p->z = z3;
}

void EllipticCurve::jAddMut(JacobPoint *p, Point *q) {
	bigint t1 = fp.square(p->z);
	bigint t2 = fp.multiply(p->z, t1);
	bigint t3 = fp.multiply(q->x, t1);
	bigint t4 = fp.multiply(q->y, t2);
	bigint t5 = fp.subtract(t3, p->x);
	bigint t6 = fp.subtract(t4, p->y);
	bigint t7 = fp.square(t5);
	bigint t8 = fp.multiply(t5, t7);
	bigint t9 = fp.multiply(p->x, t7);

	bigint x3 = fp.square(t6);
	x3 = fp.subtract(x3, fp.add(t8, fp.add(t9, t9)));

	bigint y3 = fp.multiply(t6, fp.subtract(t9, x3));
	y3 = fp.subtract(y3, fp.multiply(p->y, t8));

	bigint z3 = fp.multiply(p->z, t5);

	p->x = x3;
	p->y = y3;
	p->z = z3;
}

Point EllipticCurve::randomPoint(Random rd) {
	bigint x, y;
	do {
		x = fp.randomElement(rd);
		//temp=x^3+ax+b
		bigint temp = fp.multiply(x, x);
		       temp = fp.multiply(x, temp);
		       temp = fp.add(temp, fp.multiply(a4, x));
		       temp = fp.add(temp, a6);
	
		bigint sqr = fp.squareRoot(temp);

		if (sqr!=bigint(0)) {
			if (rand() % 2)
				y = sqr;
			else
				y = fp.negate(sqr);
			return Point(x,y);
		}
		rd.nextInt(); //TODO FIX
	} while(true);
}
