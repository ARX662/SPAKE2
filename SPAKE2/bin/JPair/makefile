objects = jpair.o uint128_t.o uint256_t.o point.o Complex.o Fp.o EllipticCurve.o TatePairing.o KeyGen.o Extract.o Encryption.o Decryption.o Random.o

all: $(objects)
	nvcc -arch=sm_20 $(objects) -o bigint

%.o: %.cu
	nvcc -x cu -arch=sm_20 -I. -dc $< -o $@

clean:
	rm -f *.o bigint
