#ifndef POINT
#define POINT

#include "uint256_t.h"
#define bigint uint256_t

// A point on elliptic curves in Affine coordinates.
// The point at Infinity is represented as (NULL, NULL).

class Point {
	public:
	   bigint x;
	   bigint y;

	   Point();
	   Point(bigint x, bigint y);

	   bool operator==(const Point &rhs) const;
	   bool operator!=(const Point &rhs) const;

	   std::string str();
};

class JacobPoint : public Point {
	public:
	   bigint z;

	   JacobPoint(bigint x, bigint y, bigint z);

	   std::string str();
};

#endif
