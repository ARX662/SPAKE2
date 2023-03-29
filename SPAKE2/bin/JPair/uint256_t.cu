#include "pair.h"
#include "uint256_t.h"

__host__ __device__ uint256_t::uint256_t() {
	UPPER = uint128_t(0);
	LOWER = uint128_t(0);
	sign = false;
}

__host__ __device__ uint256_t::uint256_t(const int &rhs) {
	UPPER = uint128_t(0);
	if(rhs < 0) {
		sign = true;
		LOWER = -rhs;
	} else {
		sign = false;
		LOWER = rhs;
	}

}

__host__ __device__ uint256_t::uint256_t(const uint256_t &rhs) {
	UPPER = rhs.UPPER;
	LOWER = rhs.LOWER;
	sign = rhs.sign;
}

__host__ __device__ uint256_t::uint256_t(const uint128_t &upper_rhs, const uint128_t &lower_rhs) {
	UPPER = upper_rhs;
	LOWER = lower_rhs;
	sign = false;
}

// Typecasts Operator implementations
__host__ __device__ uint256_t::operator bool() const {
	return (bool) (UPPER | LOWER);
}
__host__ __device__ uint256_t::operator char() const {
	return (char) LOWER;
}
__host__ __device__ uint256_t::operator int8_t() const {
	return (int8_t) LOWER;
}
__host__ __device__ uint256_t::operator uint8_t() const {
	return (uint8_t) LOWER;
}
__host__ __device__ uint256_t::operator uint128_t() const {
	return LOWER;
}

__host__ __device__ uint128_t uint256_t::upper() const {
	return UPPER;
}
__host__ __device__ uint128_t uint256_t::lower() const {
	return LOWER;
}

__host__ __device__ uint256_t uint256_t::operator=(const uint256_t &rhs) {
	UPPER = rhs.UPPER;
	LOWER = rhs.LOWER;
	return *this;
}

__host__ __device__ uint256_t uint256_t::operator+(const uint256_t &rhs) const {
	return uint256_t(UPPER + rhs.UPPER + (((LOWER + rhs.LOWER) < LOWER) ? uint128_t(1) : uint128_t(0)), LOWER + rhs.LOWER);
}

__host__ __device__ uint256_t uint256_t::operator+=(const uint256_t &rhs) {
	UPPER = rhs.UPPER + UPPER + uint128_t((LOWER + rhs.LOWER) < LOWER);
	LOWER = LOWER + rhs.LOWER;
	return *this;
}

__host__ __device__ uint256_t uint256_t::operator-(const uint256_t &rhs) const {
	return uint256_t(UPPER - rhs.UPPER - uint128_t((LOWER - rhs.LOWER) > LOWER), LOWER - rhs.LOWER);
}

__host__ __device__ uint256_t uint256_t::operator-=(const uint256_t &rhs) {
	*this = *this - rhs;
	return *this;
}

__host__ __device__ uint256_t uint256_t::operator*(const uint256_t &rhs) const {
	// split values into 4 64-bit parts
	uint128_t top[4] = {UPPER.upper(), UPPER.lower(), LOWER.upper(), LOWER.lower()};
	uint128_t bottom[4] = {rhs.upper().upper(), rhs.upper().lower(), rhs.lower().upper(), rhs.lower().lower()};
	uint128_t products[4][4];

	for(int y = 3; y > -1; y--)
		for(int x = 3; x > -1; x--)
		    products[3 - x][y] = top[x] * bottom[y];

	// initial row
	uint128_t fourth64 = products[0][3].lower();
	uint128_t third64  = products[0][2].lower() + (products[0][3].upper());
	uint128_t second64 = products[0][1].lower() + (products[0][2].upper());
	uint128_t first64  = products[0][0].lower() + (products[0][1].upper());
	// second row
	third64  += (products[1][3].lower());
	second64 += (products[1][2].lower()) + (products[1][3].upper());
	first64  += (products[1][1].lower()) + (products[1][2].upper());
	// third row
	second64 += (products[2][3].lower());
	first64  += (products[2][2].lower()) + (products[2][3].upper());
	// fourth row
	first64 += products[3][3].lower();

	// combines the values, taking care of carry over
	return uint256_t(first64 << uint128_t(64), uint128_t(0)) + uint256_t(third64.upper(), third64 << uint128_t(64)) + uint256_t(second64, uint128_t(0)) + uint256_t(fourth64);
}
__host__ __device__ uint256_t uint256_t::operator*=(const uint256_t &rhs) {
	*this = *this * rhs;
	return *this;	
}

__host__ __device__ uint256_t uint256_t::operator/(const uint256_t &rhs) const {
	return divmod(*this, rhs).first;
}
__host__ __device__ uint256_t uint256_t::operator/=(const uint256_t &rhs) {
	*this = *this / rhs;
	return *this;
}
__host__ __device__ pair_2<uint256_t, uint256_t> uint256_t::divmod(const uint256_t &lhs, const uint256_t &rhs) const {
	// Save some calculations
	if (rhs == uint256_t(1)) {
		return pair_2<uint256_t, uint256_t> (lhs, uint256_t(0));
	} else if (lhs == rhs) {
		return pair_2<uint256_t, uint256_t> (uint256_t(1), uint256_t(0));
	} else if ((lhs == uint256_t(0)) || (lhs < rhs)) {
		return pair_2<uint256_t, uint256_t> (uint256_t(0), lhs);
	}

	pair_2<uint256_t, uint256_t> qr(uint256_t(0), lhs);
	uint256_t copyd = rhs << uint256_t(lhs.bits() - rhs.bits());
	uint256_t adder = uint256_t(1) << uint256_t(lhs.bits() - rhs.bits());
	if (copyd > qr.second) {
		copyd >>= uint256_t(1);
		adder >>= uint256_t(1);
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

__host__ __device__ uint256_t uint256_t::operator%(const uint256_t &rhs) const {
	return *this - (rhs * (*this / rhs));
}
__host__ __device__ uint256_t uint256_t::operator%=(const uint256_t &rhs) {
	*this = *this % rhs;
	return *this;
}


__host__ __device__ uint16_t uint256_t::bits() const {
	uint16_t out = 0;
	if (UPPER) {
		out = 128;
		uint128_t up = UPPER;
		while (up) {
			up >>= uint128_t(1);
			out++;
		}
	} else {
		uint128_t low = LOWER;
		while (low) {
			low >>= uint128_t(1);
			out++;
		}
	}
	return out;
}

// Bitwise Operators
__host__ __device__ uint256_t uint256_t::operator&(const uint256_t &rhs) const {
	return uint256_t(UPPER & rhs.UPPER, LOWER & rhs.LOWER);
}

__host__ __device__ uint256_t uint256_t::operator<<(const uint256_t &rhs) const {
	uint128_t shift = rhs.LOWER;
	if (((bool) rhs.UPPER) || (shift >= uint128_t(256)))
		return uint256_t(0);
	else if (shift == uint128_t(128))
		return uint256_t(LOWER, uint128_t(0));
	else if (shift == uint128_t(0))
		return *this;
	else if (shift < uint128_t(128))
		return uint256_t((UPPER << shift) + (LOWER >> (uint128_t(128) - shift)), LOWER << shift);
	else if ((uint128_t(256) > shift) && (shift > uint128_t(128)))
		return uint256_t(LOWER << (shift - uint128_t(128)), uint128_t(0));
	else
		return uint256_t(0);
}
__host__ __device__ uint256_t uint256_t::operator<<=(const uint256_t &shift) {
	*this = *this << shift;
	return *this;
}

__host__ __device__ uint256_t uint256_t::operator>>(const uint256_t &rhs) const {
	uint128_t shift = rhs.LOWER;
	if (((bool) rhs.UPPER) | (shift >= uint128_t(256)))
		return uint256_t(0);
	else if (shift == uint128_t(128))
		return uint256_t(UPPER);
	else if (shift == uint128_t(0))
		return *this;
	else if (shift < uint128_t(128))
		return uint256_t(UPPER >> shift, (UPPER << (uint128_t(128) - shift)) + (LOWER >> shift));
	else if ((uint128_t(256) > shift) && (shift > uint128_t(128)))
		return uint256_t(UPPER >> (shift - uint128_t(128)));
	else
		return uint256_t(0);
}
__host__ __device__ uint256_t uint256_t::operator>>=(const uint256_t &shift) {
	*this = *this >> shift;
	return *this;
}

__host__ __device__ bool uint256_t::operator!() const {
	return !(UPPER | LOWER);
}


__host__ __device__ uint256_t uint256_t::operator|(const uint256_t &rhs) const {
	return uint256_t(UPPER | rhs.UPPER, LOWER | rhs.LOWER);
}

__host__ __device__ uint256_t uint256_t::operator|=(const uint256_t &rhs) {
	UPPER |= rhs.UPPER;
	LOWER |= rhs.LOWER;
	return *this;
}

// Comparison Operators
__host__ __device__ bool uint256_t::operator==(const uint256_t &rhs) const {
	return ((UPPER == rhs.UPPER) && (LOWER == rhs.LOWER));
}

__host__ __device__ bool uint256_t::operator!=(const uint256_t &rhs) const {
    return ((UPPER != rhs.UPPER) | (LOWER != rhs.LOWER));
}

__host__ __device__ bool uint256_t::operator>(const uint256_t &rhs) const {
	if (UPPER == rhs.UPPER)
		return (LOWER > rhs.LOWER);
	if (UPPER > rhs.UPPER)
		return true;
	return false;
}
__host__ __device__ bool uint256_t::operator>=(const uint256_t &rhs) const {
	return ((*this > rhs) | (*this == rhs));
}

__host__ __device__ bool uint256_t::operator<(const uint256_t &rhs) const {
	if (UPPER == rhs.UPPER)
		return (LOWER < rhs.LOWER);
	if (UPPER < rhs.UPPER)
		return true;
	return false;
}
__host__ __device__ bool uint256_t::operator<=(const uint256_t &rhs) const {
	return ((*this < rhs) | (*this == rhs));
}

__host__ std::string uint256_t::str(uint8_t base, const unsigned int &len) const {
	std::string out = "";
	if (!(*this))
		out = "0";
	else {
		pair_2<uint256_t, uint256_t> qr(*this, uint256_t(0));
		do {
			qr = divmod(qr.first, base);
			out = "0123456789abcdef"[(uint8_t) qr.second] + out;
		} while (qr.first);
	}
	if (out.size() < len)
		out = std::string(len - out.size(), '0') + out;

	if (sign)
		out = '-' + out; 
	return out;
}

__host__ __device__ void uint256_t::toByteArray(int8_t *in, int byteLen) {
	for(int i=0; i<byteLen; i++)
		in[(byteLen-1)-i] = (LOWER >> uint128_t(i*8));
}

__host__ __device__ void uint256_t::negate() {
	sign = !this->sign;
}
/*
__host__ __device__ void uint256_t::randomBits(int numBits, unsigned int seed) {
	srand(seed);
	int numBytes = (int) (((long) numBits+7)/8);
	int8_t randomBits[numBytes];
	if(numBytes>0) {
		for(int i=0; i<numBytes; i++)
			randomBits = (int8_t) rand();
		int excessBits = 8 * numBytes - numBits;
		randomBits[0] &= (1 << (8-excessBits)) - 1;
	}
	return randomBits;
}
*/
