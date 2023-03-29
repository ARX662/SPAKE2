#ifndef ENCRYPTION
#define ENCRYPTION

class Encryption {
	public:
		Complex c_twil;
		Point c_0, c_1, c_2;

		Encryption();
		Encryption(int sec1, TatePairing e, Complex omega, EllipticCurve g1, Point h, Point v_1, Point v_2);
};

#endif
