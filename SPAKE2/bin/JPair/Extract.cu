#include "pair.h"
#include "point.h"
#include "Fp.h"
#include "EllipticCurve.h"
#include "TatePairing.h"
#include "Extract.h"

Extract::Extract() {}

Extract::Extract(int sec1, TatePairing e, bigint t_1, bigint t_2, bigint w, EllipticCurve g1, Point g, Point h) {
	cout << endl << endl << "EXTRACT" << endl;
	cout << "a4=" << g1.a4.str() << ", a6=" << g1.a6.str() << ", opt=" << g1.opt << endl;
	cout << "mod=" << g1.fp.mod.str() << endl;

	Random r10(10);
	r = bigint((uint16_t) r10.nextInt()); /*53882*/
	rt_1t_2 = r*(t_1*t_2);

	minusOnewt_2 = w * t_2;
	minusOnewt_2.negate();
	minusOnert_2 = r * t_2;
	minusOnert_2.negate();
	minusOnewt_1 = w * t_1;
	minusOnewt_1.negate();
	minusOnert_1 = r * t_1;
	minusOnert_1.negate();

	d_0 = g1.multiply(g, rt_1t_2);	//sk

	Point d_11 = g1.multiply(g, minusOnewt_2);
	Point d_12 = g1.multiply(h, minusOnert_2);
	d_1 = g1.add(d_11, d_12);	//sk

	Point d_21 = g1.multiply(g, minusOnewt_1);
	Point d_22 = g1.multiply(h, minusOnert_1);
	d_2 = g1.add(d_21, d_22);	//sk

	cout << "d_0=" << d_0.str() << endl;
	cout << "d_1=" << d_1.str() << endl;
	cout << "d_2=" << d_2.str() << endl;
}
