#ifndef KEYGEN
#define KEYGEN

class KeyGen {
	public:
		int sec1;
		Complex omega;	
		bigint t_1, t_2, w;
		EllipticCurve g1;
		Fp gt;
		Point g, v_1, v_2;
		TatePairing e;

		KeyGen();
};

#endif
