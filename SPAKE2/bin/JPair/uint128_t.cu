#include "pair.h"
#include "uint128_t.h"

// Constructors
__host__ __device__ uint128_t::uint128_t() {
	UPPER = 0;
	LOWER = 0;
}

__host__ __device__ uint128_t::uint128_t(const uint64_t &rhs) {
	UPPER = 0;
	LOWER = rhs;
}
__host__ __device__ uint128_t::uint128_t(const uint64_t &upper_rhs, const uint64_t &lower_rhs) {
	UPPER = upper_rhs;
	LOWER = lower_rhs;
}

__host__ __device__ uint128_t::uint128_t(const uint128_t &rhs) {
	UPPER = rhs.UPPER;
	LOWER = rhs.LOWER;
}

__host__ __device__ uint64_t uint128_t::upper() const {
	return UPPER;
}

__host__ __device__ uint64_t uint128_t::lower() const {
	return LOWER;
}

//Typecast Operator
__host__ __device__ uint128_t::operator bool() const {
	return (bool) (UPPER | LOWER);
}
__host__ __device__ uint128_t::operator char() const {
	return (char) LOWER;
}
__host__ __device__ uint128_t::operator int() const {
	return (int) LOWER;
}
__host__ __device__ uint128_t::operator int8_t() const {
	return (int8_t) LOWER;
}
__host__ __device__ uint128_t::operator uint8_t() const {
	return (uint8_t) LOWER;
}
__host__ __device__ uint128_t::operator uint16_t() const {
	return (uint16_t) LOWER;
}
__host__ __device__ uint128_t::operator uint32_t() const {
	return (uint32_t) LOWER;
}
__host__ __device__ uint128_t::operator uint64_t() const {
	return (uint64_t) LOWER;
}


// RHS input args only
// Assignment Operator
__host__ __device__ uint128_t uint128_t::operator=(const uint128_t &rhs) {
	UPPER = rhs.UPPER;
	LOWER = rhs.LOWER;
	return *this;
}


// Arithmetic Operators
__host__ __device__ uint128_t uint128_t::operator+(const uint128_t &rhs) const {
	return uint128_t(UPPER + rhs.UPPER + ((LOWER + rhs.LOWER) < LOWER), LOWER + rhs.LOWER);
}
__host__ __device__ uint128_t uint128_t::operator+=(const uint128_t &rhs) {
	UPPER = rhs.UPPER + UPPER + ((LOWER + rhs.LOWER) < LOWER);
	LOWER += rhs.LOWER;
	return *this;
}

__host__ __device__ uint128_t uint128_t::operator-(const uint128_t &rhs) const {
	return uint128_t(UPPER - rhs.UPPER - ((LOWER - rhs.LOWER) > LOWER), LOWER - rhs.LOWER);
}
__host__ __device__ uint128_t uint128_t::operator-=(const uint128_t &rhs) {
	*this = *this - rhs;
	return *this;
}

__host__ __device__ uint128_t uint128_t::operator*(const uint128_t &rhs) const {
	// split values into 4 32-bit parts
	uint64_t top[4] = {UPPER >> 32, UPPER &0xffffffff, LOWER>>32, LOWER &0xffffffff};
	uint64_t bottom[4] ={rhs.UPPER >> 32, rhs.UPPER & 0xffffffff, rhs.LOWER >> 32, rhs.LOWER & 0xffffffff};
	uint64_t products[4][4];

	for(int y = 3; y > -1; y--)
		for(int x = 3; x > -1; x--)
			products[3 - x][y] = top[x] * bottom[y];

	// initial row
	uint64_t fourth32 = (products[0][3] & 0xffffffff);
	uint64_t third32  = (products[0][2] & 0xffffffff) + (products[0][3] >> 32);
	uint64_t second32 = (products[0][1] & 0xffffffff) + (products[0][2] >> 32);
	uint64_t first32  = (products[0][0] & 0xffffffff) + (products[0][1] >> 32);
	// second row
	third32  += (products[1][3] & 0xffffffff);
	second32 += (products[1][2] & 0xffffffff) + (products[1][3] >> 32);
	first32  += (products[1][1] & 0xffffffff) + (products[1][2] >> 32);
	// third row
	second32 += (products[2][3] & 0xffffffff);
	first32  += (products[2][2] & 0xffffffff) + (products[2][3] >> 32);
	// fourth row
	first32 += (products[3][3] & 0xffffffff);

	// combines the values, taking care of carry over
	return uint128_t(first32 << 32, 0) + uint128_t(third32 >> 32, third32 << 32) + uint128_t(second32, 0) + uint128_t(fourth32);
}

__host__ __device__ uint128_t uint128_t::operator/(const uint128_t &rhs) const {
	return divmod(*this, rhs).first;
}
__host__ __device__ uint128_t uint128_t::operator/=(const uint128_t &rhs) {
	*this = *this / rhs;
	return *this;
}


__host__ __device__ uint128_t uint128_t::operator%(const uint128_t &rhs) const {
	return *this - (rhs * (*this / rhs));
}
__host__ __device__ uint128_t uint128_t::operator%=(const uint128_t &rhs) {
	*this = *this % rhs;	
	return *this;
}



__host__ __device__ uint8_t uint128_t::bits() const{
	uint8_t out = 0;
	if (UPPER){
		out = 64;
		uint64_t up = UPPER;
		while (up) {
			up >>= 1;
			out++;
		}
	} else {
		uint64_t low = LOWER;
		while (low) {
			low >>= 1;
			out++;
		}
	}
	return out;
}



// Bitwise Operators
__host__ __device__ uint128_t uint128_t::operator&(const uint128_t &rhs) const {
	return uint128_t(UPPER & rhs.UPPER, LOWER & rhs.LOWER);
}

__host__ __device__ uint128_t uint128_t::operator<<(const uint128_t &rhs) const {
	uint64_t shift = rhs.LOWER;
	if (((bool) rhs.UPPER) || (shift >= 128))
		return uint128_t(0);
	else if (shift == 64)
		return uint128_t(LOWER, 0);
	else if (shift == 0)
		return *this;
	else if (shift < 64)
		return uint128_t((UPPER << shift) + (LOWER >> (64 - shift)), LOWER << shift);
	else if ((128 > shift) && (shift > 64))
		return uint128_t(LOWER << (shift - 64), 0);
	else
		return uint128_t(0);
}
__host__ __device__ uint128_t uint128_t::operator<<=(const uint128_t &rhs) {
	*this = *this << rhs;
	return *this;
}


__host__ __device__ uint128_t uint128_t::operator>>(const uint128_t &rhs) const {
	uint64_t shift = rhs.LOWER;
	if (((bool) rhs.UPPER) || (shift >= 128))
		return uint128_t(0);
	else if (shift == 64)
		return uint128_t(0, UPPER);
	else if (shift == 0)
		return *this;
	else if (shift < 64)
		return uint128_t(UPPER >> shift, (UPPER << (64 - shift)) + (LOWER >> shift));
	else if ((128 > shift) && (shift > 64))
		return uint128_t(0, (UPPER >> (shift - 64)));
	else
		return uint128_t(0);
}
__host__ __device__ uint128_t uint128_t::operator>>=(const uint128_t &rhs) {
	*this = *this >> rhs;
	return *this;
}

__host__ __device__ bool uint128_t::operator!() const {
	return !(bool) (UPPER | LOWER);
}

__host__ __device__ uint128_t uint128_t::operator|(const uint128_t & rhs) const {
	return uint128_t(UPPER | rhs.UPPER, LOWER | rhs.LOWER);
}
__host__ __device__ uint128_t uint128_t::operator|=(const uint128_t &rhs) {
	UPPER |= rhs.UPPER;
	LOWER |= rhs.LOWER;
	return *this;
}


// Comparison Operators
__host__ __device__ bool uint128_t::operator==(const uint128_t &rhs) const{
	return ((UPPER == rhs.UPPER) && (LOWER == rhs.LOWER));
}

__host__ __device__ bool uint128_t::operator!=(const uint128_t &rhs) const {
	return ((UPPER != rhs.UPPER) | (LOWER != rhs.LOWER));
}

__host__ __device__ bool uint128_t::operator>(const uint128_t &rhs) const {
	if (UPPER == rhs.UPPER)
		return (LOWER > rhs.LOWER);
	return (UPPER > rhs.UPPER);
}
__host__ __device__ bool uint128_t::operator>=(const uint128_t &rhs) const {
	return ((*this > rhs) | (*this == rhs));
}

__host__ __device__ bool uint128_t::operator<(const uint128_t &rhs) const {
	if (UPPER == rhs.UPPER)
		return (LOWER < rhs.LOWER);
	return (UPPER < rhs.UPPER);
}
__host__ __device__ bool uint128_t::operator<=(const uint128_t &rhs) const {
	return ((*this < rhs) | (*this == rhs));
}


__host__ __device__ pair_2<uint128_t, uint128_t> uint128_t::divmod(const uint128_t &lhs, const uint128_t &rhs) const {
	// Save some calculations
	if (rhs == uint128_t(1)) {
		return pair_2<uint128_t, uint128_t> (lhs, uint128_t(0));
	} else if(lhs == rhs) {
		return pair_2<uint128_t, uint128_t> (uint128_t(1), uint128_t(0));
	} else if ((lhs == uint128_t(0)) || (lhs < rhs)) {
		return pair_2<uint128_t, uint128_t> (uint128_t(0), lhs);
	}

	pair_2<uint128_t, uint128_t> qr(uint128_t(0), lhs);
	uint128_t copyd = rhs << (uint128_t) (lhs.bits() - rhs.bits());
	uint128_t adder = uint128_t(1) << (uint128_t) (lhs.bits() - rhs.bits());
	if (copyd > qr.second) {
		copyd >>= uint128_t(1);
		adder >>= uint128_t(1);
	}
	while (qr.second >= rhs) {
		if (qr.second >= copyd) {
			qr.second -= copyd;
			qr.first |= adder;
		}
		copyd >>= uint128_t(1);
		adder >>= uint128_t(1);
	}
	return qr;
}

// Formatted print
__host__ string uint128_t::str(uint8_t base, const unsigned int &len) const {
	string out = "";
	if (!(*this))
		out = "0";
	else {
		pair_2<uint128_t, uint128_t> qr(*this, uint128_t(0));
		do {
			qr = divmod(qr.first, base);
			out = "0123456789abcdef"[(uint8_t) qr.second] + out;
		} while (qr.first);
	}
	if (out.size() < len)
		out = std::string(len - out.size(), '0') + out;
	return out;
}

__host__ __device__ void uint128_t::toByteArray(uint8_t *in) {
	for(int i=0; i<8; i++)
		in[15-i] = (LOWER >> (i*8));
	for(int i=16; i<16; i++)
		in[7-i] = (UPPER >> (i*8));
}
