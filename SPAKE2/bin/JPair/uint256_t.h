#ifndef __UINT256_T__
#define __UINT256_T__

#include "uint128_t.h"
#include <stdint.h>

class uint256_t {
    private:
	uint128_t UPPER, LOWER;

    public:
	bool sign;

	// Constructors
	__host__ __device__ uint256_t();
	__host__ __device__ uint256_t(const int &rhs);
	__host__ __device__ uint256_t(const uint256_t &rhs);
	__host__ __device__ uint256_t(const uint128_t &upper_rhs, const uint128_t &lower_rhs);

	template <typename T> __host__ __device__ uint256_t(const T &rhs) {
		UPPER = uint128_t(0);
		LOWER = rhs;
	}
	
	
        // Get private values
        __host__ __device__ uint128_t upper() const;
        __host__ __device__ uint128_t lower() const;

	//Typecast Operator
	__host__ __device__ operator bool() const;
	__host__ __device__ operator char() const;
	__host__ __device__ operator int8_t() const;
	__host__ __device__ operator uint8_t() const;
	__host__ __device__ operator uint128_t() const;

	// RHS input args only
	// Assignment Operator
	__host__ __device__ uint256_t operator=(const uint256_t &rhs);

	// Arithmetic Operators
	__host__ __device__ uint256_t operator+(const uint256_t &rhs) const;
	__host__ __device__ uint256_t operator+=(const uint256_t &rhs);
	__host__ __device__ uint256_t operator-(const uint256_t &rhs) const;
	__host__ __device__ uint256_t operator-=(const uint256_t &rhs);
	__host__ __device__ uint256_t operator*(const uint256_t &rhs) const;
	__host__ __device__ uint256_t operator*=(const uint256_t &rhs);
	__host__ __device__ uint256_t operator/(const uint256_t &rhs) const;
	__host__ __device__ uint256_t operator/=(const uint256_t &rhs);
	__host__ __device__ uint256_t operator%(const uint256_t &rhs) const;
	__host__ __device__ uint256_t operator%=(const uint256_t &rhs);

	// Bitwise Operators
	__host__ __device__ uint256_t operator&(const uint256_t & rhs) const;
	__host__ __device__ uint256_t operator<<(const uint256_t &shift) const;
	__host__ __device__ uint256_t operator>>(const uint256_t &shift) const;
        __host__ __device__ uint256_t operator<<=(const uint256_t &shift);
        __host__ __device__ uint256_t operator>>=(const uint256_t &shift);
	__host__ __device__ bool operator!() const;
	__host__ __device__ uint256_t operator|(const uint256_t &rhs) const;
	__host__ __device__ uint256_t operator|=(const uint256_t &rhs);
	
	// Comparison Operators
	__host__ __device__ bool operator==(const uint256_t &rhs) const;
	__host__ __device__ bool operator!=(const uint256_t & rhs) const;
	__host__ __device__ bool operator>(const uint256_t &rhs) const;
        __host__ __device__ bool operator>=(const uint256_t &rhs) const;
        __host__ __device__ bool operator<(const uint256_t &rhs) const;
        __host__ __device__ bool operator<=(const uint256_t &rhs) const;

	// Get bitsize of value
        __host__ __device__ uint16_t bits() const;
	__host__ __device__ void toByteArray(int8_t *in, int byteLen);

        // Get string representation of value
        __host__ std::string str(uint8_t base = 10, const unsigned int &len = 0) const;

	__host__ __device__ void negate();

//	__host__ __device__ uint256_t randomBits(int numBits, unsigned int seed);

    private:
	__host__ __device__ pair_2<uint256_t, uint256_t> divmod(const uint256_t &lhs, const uint256_t &rhs) const;
};

#endif

