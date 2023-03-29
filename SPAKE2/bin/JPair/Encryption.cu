#include "pair.h"
#include "point.h"
#include "Complex.h"
#include "Fp.h"
#include "EllipticCurve.h"
#include "TatePairing.h"
#include "Encryption.h"

Encryption::Encryption() {}

Encryption::Encryption(int sec1, TatePairing e, Complex omega, EllipticCurve g1, Point h, Point v_1, Point v_2) {
	cout << endl << endl << "ENCRYPTION" << endl;
	Random r0(0);
	int s_i = (uint16_t) r0.nextInt();
		  /* 24756*/
	Random r1(1);     
	int s_1_i = (uint16_t) r1.nextInt();
		    /* 29653*/

	bigint   s(s_i);	
	bigint s_1(s_1_i);

	c_twil = e.gt.power(omega, s);
	c_0 = g1.multiply(h, s);
	c_1 = g1.multiply(v_1, s_i - s_1_i);
	c_2 = g1.multiply(v_2, s_1);
}
