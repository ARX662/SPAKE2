#ifndef EXTRACT
#define EXTRACT

class Extract {
	public:
		bigint r, rt_1t_2;
		Point d_0, d_1, d_2;
		bigint minusOnewt_2, minusOnert_2;
		bigint minusOnewt_1, minusOnert_1;

		Extract();
		Extract(int sec1, TatePairing e, bigint t_1, bigint t_2, bigint w, 
			EllipticCurve g1, Point g, Point h);
};

#endif
