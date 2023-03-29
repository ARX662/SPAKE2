#include "pair.h"
#include "point.h"
#include "Complex.h"
#include "Fp.h"
#include "EllipticCurve.h"
#include "TatePairing.h"
#include "Decryption.h"

Decryption::Decryption() {}

Decryption::Decryption(TatePairing e, Complex ciphertext, Point c_0, Point c_1, Point c_2, Point d_0, Point d_1, Point d_2, Fp gt) {
	cout << endl << endl << "DECRYPTION" << endl;

	Complex e_0 = e.compute(c_0, d_0);
	cout << "e_0=" << e_0.str() << endl << endl;

	Complex e_1 = e.compute(c_1, d_1);
	cout << "e_1=" << e_1.str() << endl << endl;

	Complex e_2 = e.compute(c_2, d_2);
	cout << "e_2=" << e_2.str() << endl << endl;

	out = gt.multiply(ciphertext, gt.multiply(e_0, gt.multiply(e_1, e_2)));
	cout << "res=" << out.str() << endl;
}
