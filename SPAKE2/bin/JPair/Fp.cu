#include "Random.h"
#include "pair.h"
#include "point.h"
#include "Fp.h"
#include "Complex.h"

Fp::Fp() {}

Fp::Fp(bigint mod) {
	this->mod = mod;
}


bigint Fp::add(bigint a, bigint b) {
	bigint res = a + b;
	res %= mod;
	return res;
}

bigint Fp::subtract(bigint a, bigint b) {
	bigint res;

	if (a>b)
            res = a - b;
	else 
	    res = b - a;
        res %= mod;

	if (a<b)
	   res = mod - res;

	return res;
}

bigint Fp::multiply(bigint a, bigint b) {
	bigint res(0);
	bigint addend = b;

	while (a) {
		bigint bit = a & bigint(1);
		if (bit == bigint(1)) {
			res += addend;
			res %= mod;
		}
		addend += addend;
		addend %= mod;
		a >>= 1;
	}
	return res;
}

bigint Fp::divide(bigint a, bigint b) {
	return (multiply(a, inverse(b))) % mod;
}

bigint Fp::square(bigint n) {
	return multiply(n, n);
}


bigint Fp::power(bigint x, bigint y) {
	if (y == bigint(0))
		return 1;
	bigint p = power(x, y/bigint(2)) % mod;
	p = multiply(p, p);
	return (y % bigint(2) == bigint(0)) ? p : multiply(x, p);
}

bigint Fp::inverse(bigint a) {
	return power(a, mod-bigint(2));
}

bigint Fp::negate(bigint a) {
	a = a % mod;
	return mod - a;
}


JacobPoint Fp::aToJ(Point p) {
	return JacobPoint(p.x, p.y, bigint(1));
}

Point Fp::jToA(JacobPoint p) {
	if(p.x==bigint(NULL) && p.y==bigint(NULL))
		return Point();

	bigint zInverse = inverse(p.z);
	bigint square = multiply(zInverse, zInverse);
	// x = X/Z^2
	bigint x = multiply(p.x, square);
	// y = Y/Z^3
	bigint y = multiply(p.y, multiply(square, zInverse));
	return Point(x, y);
}


bigint Fp::randomElement(Random rd) {
	bigint rdm(rd.nextInt()); //, rd.nextInt()); //TODO Test
	while(rdm >= mod)
		rdm>>=bigint(1);
	cout << "=" << rdm.str() << endl;
	return rdm;
}

bigint Fp::squareRoot(bigint val) {
	bigint p = mod;
	bigint g = val;
	//g==0 || g==1
	if (g==bigint(0) || g==bigint(1))
		return g;
	// Divide and Remainder (4)
	bigint result[2];
	result[0] = p / bigint(4);
	result[1] = p % bigint(4);
	//p=3 mod 4 i.e. p=4k+3, output val^{k+1} mod p
	if (result[1]==bigint(3)) {
		bigint k = result[0];
		bigint z = power(g,(k+bigint(1))) % p;
		if (square(z) == g)
			return z;
		else
			return NULL;
	}
	// Divide and Remainder (8)
	result[0] = p / bigint(8);
	result[1] = p % bigint(8);
	bigint k = result[0];
	bigint remainder = result[1];
	//p=5 mod 8 i.e. p=8k+5
	if (remainder == bigint(5)) {
		//gamma=(g2)^k mod p
		bigint g2 = g*bigint(2);
		bigint gamma = power(g2,k) % p;
		//i=g2*gamma^2 mod p
		bigint i = g2*(power(gamma,bigint(2))) % p;
		//output g*gamma(i-1) mod p
		bigint z = (g*gamma*(i-bigint(1))) % p;
		if (square(z) == g)
			return z;
		else
			return NULL;
	} else if (remainder == bigint(1)) {
		bigint q=g;
		do {
			bigint p = randomElement(Random(0)); //TODO Poss 1.
			bigint k = (p + bigint(1) / bigint(2));
			
			bigint re[2];
			LucasSequence(p,q,k,re);

			bigint v = re[0];
			bigint q0 = re[1];

			bigint z = (v / bigint(2) % p);

			if (square(z) == g)
				return z;
			if (q0 > bigint(1))
				if(q0 < p-bigint(1))
					return NULL;
		} while (true);
	}
	return NULL;
}

bigint* Fp::LucasSequence(bigint p, bigint q, bigint k, bigint re[]) {
	cout << "Fp: LucusSequence" << endl;
	bigint v0(2);
	bigint v1=p;
	bigint q0(1);
	bigint q1(1);

	int r = k.bits()-1;

	for(int i=r; i>=0; i--) {
		//q0=q0q1 mof n
		q0 = multiply(q0, q1);
		if((k & bigint(1<<i)) != bigint(0)) {
			//q1=q0q mod n
			q1 = multiply(q0, q);
			//v0=v0v1-pq0 mod n
			v0 = multiply(v0,v1);
			v0 = subtract(v0, multiply(p, q0));
			//v1=v1^2-2q1 mod n
			v1 = square(v1);
			v1 = subtract(v1, multiply(q1, 2));
		} else{
			//q1=q0
			q1=q0;
			//v1=c0c1-pq0 mod n
			v1 = multiply(v0,v1);
			v1 = subtract(v1, multiply(p,q0));
			//v0=v0^2-2q0 mod n
			v0 = square(v0);
			v0 = subtract(v0, multiply(q0, 2));
		}
	}
	re[0]=v0;
	re[1]=q0;
	return re;
}

Complex Fp::square(Complex p) {
	//1 or 0
	if((p.real==bigint(0) && p.imag==bigint(0)) || (p.real==bigint(1) && p.imag==bigint(0)))
		return p;

	//Real number
	if(p.imag == bigint(0)) {
		bigint newReal = multiply(p.real, p.real);
		return Complex(*this, newReal, bigint(0));
	}
	//Imaginary
	if(p.real == bigint(0)) {
		bigint newReal = multiply(p.imag, p.imag);
		newReal = negate(newReal);
		return Complex(*this, newReal, bigint(0));
	}

	//(a+bi)^2=(a+b)(a-b)+2abi
	bigint newReal;
	newReal = multiply(p.real + p.imag, subtract(p.real, p.imag)); 
	
	bigint newImag = multiply(p.real + p.real, p.imag);
	Complex p2 = Complex(*this, newReal, newImag);

	return p2;
}

Complex Fp::multiply(Complex a, Complex b) {
	if((a.real==bigint(0) && a.imag==bigint(0)) || (b.real==bigint(0) && b.imag==bigint(0)))
		return Complex(*this, bigint(0), bigint(0));

	//(a+bi)*(c+di)=ac+adi+bci-bd
	//(a+ib)(c+id) = ac -bd + i[(a+b)(c+d)-ac-bd]
	bigint ac = multiply(a.real, b.real);
	bigint bd = multiply(a.imag, b.imag);

	bigint newReal = subtract(ac, bd);
	bigint newImag = a.real + a.imag;

	newImag = multiply(newImag, (b.real + b.imag));
	newImag = subtract(newImag, ac);
	newImag = subtract(newImag, bd);

	Complex ab = Complex(*this, newReal, newImag);
	
	return ab;
}

Complex Fp::power(Complex &u, bigint k) {
	Complex u2 = square(u);
	Complex t[16];
	t[0] = u;

	for(int i=1; i<16; i++)
		t[i] = multiply(u2, t[i-1]);
	//left to right method -with windows

	int nb = k.bits();

	int nbw[1];
	int nzs[1];
	int n;
	if(nb>1)
		for(int i=nb-2;i>=0;) {
			n = window(k, i, nbw, nzs, 5);
			for(int j=0; j<nbw[0]; j++)
				u=square(u);
			if(n>0)
				u=multiply(u, t[n/2]);
			i-=nbw[0];
			if(nzs[0]!=0) {
				for(int j=0; j<nzs[0]; j++)
					u=square(u);
				i-=nzs[0];
			}
		}
	
	return u;
}

int Fp::window(bigint x, int i, int* nbs, int* nzs, int w) {
	int j, r;
	nbs[0]=1;
	nzs[0]=0;
	
	int byteLen = x.bits()/8+1;
	int8_t xa[byteLen];
	x.toByteArray(xa, byteLen);

	//check for leading 0 bit
	if(!getBitByDegree(i, xa, byteLen))
		return 0;

	//adjust window if not enough bits left
	if(i-w+1<0)
		w=i+1;
	r=1;
	for(j=i-1; j>i-w; j--) {
		nbs[0]++;
		r*=2;
		if(getBitByDegree(j, xa, byteLen))
			r+=1;
		if(r%4==0) {
			r/=4;
			nbs[0]-=2;
			nzs[0]=2;
			break;
		}
	}
	if(r%2==0) {
		r/=2;
		nzs[0]=1;
		nbs[0]--;
	}

	return r;
}

bool Fp::getBitByDegree(int deg, int8_t* ba, int bits) {
	bits *= 8;

	if(deg<0 || deg>bits)
		return false;

	int index =bits-deg-1;
	int byteIndex=index/8;
	int bitIndex =index%8;

	int8_t b=ba[byteIndex];
	int8_t b2=(int8_t)(b<<bitIndex);

	return b2<0;
}
