#include "Random.h"
#include "pair.h"
#include "point.h"
#include "Fp.h"
#include "Complex.h"
#include "EllipticCurve.h"
#include "TatePairing.h"

#include <stdio.h>

TatePairing::TatePairing() {}

TatePairing::TatePairing(EllipticCurve curve, bigint groupOrder, bigint coFactor) {
	this->ec = curve;
	this->field = ec.fp;

	this->order = groupOrder;
	this->finalExponent = (field.mod + bigint(1)) / groupOrder;

	this->cof = coFactor;
	this->twisted = twist(curve);
	this->gt = field;
}

EllipticCurve TatePairing::twist(EllipticCurve e) {
	Fp f = e.fp;
	bigint b = f.negate(e.a6);
	return EllipticCurve(f, e.a4, b);
}

Point TatePairing::RandomPointInG1(Random rd) {
	Point p;
	do {
		p = ec.randomPoint(rd);
		p = ec.multiply(p, cof);
	} while(ec.multiply(p, order) != Point());

	return p;
}

/*
Point TatePairing::RandomPointInG1() {
	uint128_t low_x(3157993585693054655, 953210005121682535);
	uint128_t up_x(0, 0);
	bigint x(up_x, low_x);

	uint128_t low_y(1997117534300516058, 4167166996755168025);
	uint128_t up_y(0, 0);
	bigint y(up_y, low_y);

	Point p = Point(x, y);

	return p;
}

Point TatePairing::RandomPointInG1_2() {
	uint128_t up(0, 0);
	uint128_t low_x(5938001448379886511, 11025142976960672264);
	uint128_t low_y(4487756422352902290, 4496261878358631838);
	bigint x(up, low_x);
	bigint y(up, low_y);

	Point p = Point(x, y);
	return p;
}
*/

// Compute e(P,Q)
Complex TatePairing::compute(Point P, Point Q) {
	cout << "TatePairing: compute()" << endl;
	Complex f(field, bigint(1));

	JacobPoint V = field.aToJ(P);
	Point nP = ec.negate(P);

	uint8_t r[this->order.bits()];
	naf(r, order, (uint8_t) 2);

	Complex u;
	for (int i=this->order.bits()-2; i>=0; i--) {
		u = encDbl(V,Q);
		f = gt.multiply(gt.square(f), u);

		if(signed(r[i]) == 1) {
			u = encAdd(V,P,Q);
			f = gt.multiply(f, u);
		}
		if(signed(r[i]) == -1) {
			u = encAdd(V, nP, Q);
			f = gt.multiply(f, u);
		}
	}

	f = f.conjugate().divide(f);

	return gt.power(f, finalExponent);
}



// Windowed naf form of bigint k, where w is the window size
void TatePairing::naf(uint8_t *r, bigint k, uint8_t w) {
	// The window NAD is at most 1 element longer than the binary representation of the integer k.	
	uint8_t wnaf[k.bits()+1];

	//2^width as short and bigint
	short pow2wB = (short)(1<<w);
	bigint pow2wBI = bigint(pow2wB);

	int i=0;
	int length=0;

	while(k>bigint(0)) {
		if((k&bigint(1<<0))!=bigint(0)) {
			bigint remainder = k % pow2wBI;
			if((remainder & bigint(1<<(w-1))) != bigint(0)) {
				wnaf[i] = (uint8_t) remainder - pow2wB;
			} else {
				wnaf[i] = (uint8_t) remainder;
			}
			k = k - bigint(wnaf[i]);
			length = i;
		} else {
			wnaf[i] = 0;
		}
		k = k>>bigint(1);
		i++;
	}

	length++;

	for(int i=0; i<length; i++)
		r[i]=wnaf[i];
}


// Used by TatePairing, point doubling in Jacobian coordinates, and return the value of f
Complex TatePairing::encDbl(JacobPoint &P, Point Q) {
	bigint x = P.x;
	bigint y = P.y;
	bigint z = P.z;

	//t1=y^2
	bigint t1 = field.square(y);
	//t2=4xt1
	bigint t2 = field.multiply(x,t1);
	t2 = field.add(t2,t2);
	t2 = field.add(t2,t2);

	//t3=8t1^2
	bigint t3 = field.square(t1);
	t3 = field.add(t3,t3);
	t3 = field.add(t3,t3);
	t3 = field.add(t3,t3);

	//t4=z^2
	bigint t4 = field.square(z);

	bigint t5;
	if(ec.opt) {
		t5 = field.multiply(field.subtract(x,t4), field.add(x,t4));
		t5 = field.add(t5, field.add(t5,t5));
	} else {
		//t5=3x^2+aZ^4
		t5 = field.square(x);
		t5 = field.add(t5, field.add(t5,t5));
		t5 = field.add(t5, field.multiply(ec.a4, field.square(t4)));
	}
	//x3=t5^2-2t2
	bigint x3 = field.square(t5);
	x3 = field.subtract(x3, field.add(t2,t2));

	//y3=t5(t2-x3)-t3
	bigint y3 = field.multiply(t5, field.subtract(t2,x3));
	y3 = field.subtract(y3, t3);

	//z3=2y1z1
	bigint z3 = field.multiply(y,z);
	z3 = field.add(z3,z3);

	P.x = x3;
	P.y = y3;
	P.z = z3;

	//z3t4yQi-(2t1-t5(t4Xq+x1))
	bigint real = field.multiply(t4, Q.x);
	real = field.add(real, x);
	real = field.multiply(t5, real);
	real = field.subtract(real, t1);
	real = field.subtract(real, t1);

	bigint imag = field.multiply(z3,t4);
	imag = field.multiply(imag, Q.y);

	return Complex(field, real, imag);
}

//Used by Tate Pairing to add two points, saving the result in the first argument and returning the value of f
Complex TatePairing::encAdd(JacobPoint &A, Point P, Point Q) {
	bigint x1 = A.x;
	bigint y1 = A.y;
	bigint z1 = A.z;

	bigint x = P.x;
	bigint y = P.y;

	//t1=z1^2
	bigint t1 = field.square(z1);
	//t2=z1t1
	bigint t2 = field.multiply(z1,t1);
	//t3=xt1
	bigint t3 = field.multiply(x,t1);
	//t4=yt2
	bigint t4 = field.multiply(y,t2);
	//t5=t3-x1
	bigint t5 = field.subtract(t3,x1);
	//t6=t4-y1
	bigint t6 = field.subtract(t4,y1);
	//t7=t5^2
	bigint t7 = field.square(t5);
	//t8=t5t7
	bigint t8 = field.multiply(t5,t7);
	//t9=x1t7
	bigint t9 = field.multiply(x1,t7);

	//x3=t6^2-(t8+2t9)
	bigint x3 = field.square(t6);
	x3 = field.subtract(x3, field.add(t8, field.add(t9,t9)));

	//y3=t6(t9-x3)-y1t8
	bigint y3 = field.multiply(t6, field.subtract(t9,x3));
	y3 = field.subtract(y3, field.multiply(y1,t8));

	//z3=z1t5
	bigint z3 = field.multiply(z1, t5);

	A.x = x3;
	A.y = y3;
	A.z = z3;

	//z3yqi -(z3T-t6(xq+x))
	bigint imag = field.multiply(z3, Q.y);
	
	bigint real = field.add(Q.x, x);
	real = field.multiply(real, t6);
	real = field.subtract(real, field.multiply(z3,y));

	return Complex(field, real, imag);
}

