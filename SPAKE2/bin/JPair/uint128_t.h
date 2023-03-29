#ifndef __UINT128_T__
#define __UINT128_T__

#include <stdint.h>
#include <stdexcept>
#include <iostream>
using namespace std;

class uint128_t {
    private:
	uint64_t UPPER, LOWER;

    public:
	// Constructors
	__host__ __device__ uint128_t();
	__host__ __device__ uint128_t(const uint64_t &rhs);
	__host__ __device__ uint128_t(const uint64_t &upper_rhs, const uint64_t &lower_rhs);
	__host__ __device__ uint128_t(const uint128_t &rhs);

	// Get Private Values
	__host__ __device__ uint64_t upper() const;
	__host__ __device__ uint64_t lower() const;

	//Typecast Operator
	__host__ __device__ operator bool() const;
	__host__ __device__ operator char() const;
	__host__ __device__ operator int() const;
	__host__ __device__ operator int8_t() const;
	__host__ __device__ operator uint8_t() const;
	__host__ __device__ operator uint16_t() const;
	__host__ __device__ operator uint32_t() const;
	__host__ __device__ operator uint64_t() const;

	// RHS input args only
	// Assignment Operator
	__host__ __device__ uint128_t operator=(const uint128_t &rhs);

	// Arithmetic Operators
	__host__ __device__ uint128_t operator+(const uint128_t &rhs) const;
	__host__ __device__ uint128_t operator+=(const uint128_t &rhs);
	__host__ __device__ uint128_t operator-(const uint128_t &rhs) const;
	__host__ __device__ uint128_t operator-=(const uint128_t &rhs);
	__host__ __device__ uint128_t operator*(const uint128_t &rhs) const;
	__host__ __device__ uint128_t operator/(const uint128_t &rhs) const;
	__host__ __device__ uint128_t operator/=(const uint128_t &rhs);
	__host__ __device__ uint128_t operator%(const uint128_t &rhs) const;
	__host__ __device__ uint128_t operator%=(const uint128_t &rhs);

	__host__ __device__ uint8_t bits() const;
	__host__ __device__ void toByteArray(uint8_t *in);

	// Bitwise Operators
	__host__ __device__ uint128_t operator&(const uint128_t & rhs) const;
	__host__ __device__ uint128_t operator<<(const uint128_t &rhs) const;
	__host__ __device__ uint128_t operator<<=(const uint128_t &rhs);
	__host__ __device__ uint128_t operator>>(const uint128_t &rhs) const;
	__host__ __device__ uint128_t operator>>=(const uint128_t &rhs);
	__host__ __device__ bool operator!() const;
	__host__ __device__ uint128_t operator|(const uint128_t &rhs) const;
	__host__ __device__ uint128_t operator|=(const uint128_t &rhs);

	// Comparison Operators
	__host__ __device__ bool operator==(const uint128_t &rhs) const;
	__host__ __device__ bool operator!=(const uint128_t & rhs) const;
	__host__ __device__ bool operator>(const uint128_t &rhs) const;
	__host__ __device__ bool operator>=(const uint128_t &rhs) const;
	__host__ __device__ bool operator<(const uint128_t &rhs) const;
	__host__ __device__ bool operator<=(const uint128_t &rhs) const;

	// Formatted print
	__host__ string str(uint8_t base=10, const unsigned int &len=0) const;

    private:
	__host__ __device__ pair_2<uint128_t, uint128_t> divmod(const uint128_t &lhs, const uint128_t &rhs) const;

};
#endif

