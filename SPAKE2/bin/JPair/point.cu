#include "pair.h"
#include "point.h"

Point::Point() {
	this->x = NULL;
	this->y = NULL;
}

Point::Point(bigint x, bigint y) {
	this->x = x;
	this->y = y;
}

bool Point::operator==(const Point &rhs) const {
	return x==x && y==y;
}

bool Point::operator!=(const Point &rhs) const {
	return x!=x && y!=y;
}

std::string Point::str() {
	return "[" + x.str() + "," + y.str() + "]";
}

JacobPoint::JacobPoint(bigint x, bigint y, bigint z) : Point(x, y) {
	this->z = z;
}

std::string JacobPoint::str() {
	return "[" + x.str() + "," + y.str() + "," + z.str() + "]";
}
