#include "pair.h"
#include "point.h"
#include "Complex.h"
#include "Fp.h"
#include "EllipticCurve.h"
#include "TatePairing.h"
#include "KeyGen.h"

KeyGen::KeyGen() {
	cout << "ExtractKeys..." << endl;

	sec1 = 16;

	// prime = 122365946062993565078966846847292526143
	uint128_t low(6633471227987084073, 5293657587167514175);
	uint128_t up(0, 0);
	bigint p(up, low);

	Fp field(p);
	cout << "Modular ring is created by "<< p.bits() <<" bits prime number: p = "<< p.str() << endl;
	cout << "mod=" << p.str() << "(" << &p << ")" << endl;

	bigint one(1);
	bigint zero(0);
	EllipticCurve ec(field, one, zero);
	cout << "Elliptic curve: y^2 = x^3 + x  (Created using above modular ring)" << endl;
	cout << "a4=" << ec.a4.str() << ", a6=" << ec.a6.str() << ", opt=" << ec.opt << endl;

	// group order = 40961
	bigint r(40961);

	// cofactor = 2987376921046692343423423423434304
	uint128_t low2(161946027391594, 1008460575635918400);
	bigint cof(up, low2);
	cout << "Group Order = " << r.str() << endl;
	cout << "Cofactor = " << cof.str() << endl;
	

	// Initialise the pairing
	e = TatePairing(ec, r, cof);
	cout << "Tate Pairing initialised" << endl;


	// get P, which is a random point in group G1
	g = e.RandomPointInG1(Random(0));
	cout << "Random Point on G1: X=" << g.x.str() << ", Y= "<< g.y.str() << endl;

	// The curve on which G1 is defined
	g1 = ec;
	gt = e.gt;

	Random r1(1);  
	t_1 = bigint((uint16_t) r1.nextInt());/*14706*/
	Random r2(2);    
	t_2 = bigint((uint16_t) r2.nextInt());/*19603*/ //msk
	Random r3(3);    
	  w = bigint((uint16_t) r3.nextInt());/*29653*/

	cout << "t_1 = " << t_1.str() << endl;
	cout << "t_2 = " << t_2.str() << endl;
	cout << "w   = " <<   w.str() << endl;

	bigint wt_1t_2 = w*(t_1*t_2);
	cout << "wt_1t_2=" << wt_1t_2.str() << endl;

	Complex epp = e.compute(g, g);
	cout << "epp = " << epp.str() << endl; 

	omega = gt.power(epp, wt_1t_2); //pk
	cout << "omega = " << omega.str() << endl;

	v_1 = g1.multiply(g, t_1);
	v_2 = g1.multiply(g, t_2);
	cout << "v_1 = (x=" << v_1.x.str() << ", y=" << v_1.y.str() << ")" << endl;
	cout << "v_2 = (x=" << v_2.x.str() << ", y=" << v_2.y.str() << ")" << endl;
	cout << "Keys Generated" << endl << endl;
}
