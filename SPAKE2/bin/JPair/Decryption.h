#ifndef DECRYPTION
#define DECRYPTION

class Decryption {
	public:
		Complex out;

		Decryption();
		Decryption(TatePairing e, Complex ciphertext, Point c_0, Point c_1, Point c_2, Point d_0, Point d_1, Point d_2, Fp gt);
};

#endif
