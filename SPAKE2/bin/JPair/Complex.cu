#include "pair.h"
#include "point.h"
#include "Fp.h"
#include "Complex.h"

Complex::Complex() {}

Complex::Complex(Fp field, bigint real) {
	this->field = field;
	this->real = real;
	this->imag = bigint(0);
}

Complex::Complex(Fp field, bigint real, bigint imag) {
	this->field = field;
	this->real = real;
	this->imag = imag;
}

Complex Complex::conjugate() {
	Complex c = Complex(field, real, field.negate(imag));
	return c;
}

Complex Complex::multiply(Complex p) {
	//(a+bi)*(c+di)=ac+adi+bci-bd
	//(a+ib)(c+id) = ac-bd + i[(a+b)(c+d)-ac-bd)]
	bigint ac = field.multiply(real, p.real);
	bigint bd = field.multiply(imag, p.imag);
	
	bigint newReal = field.subtract(ac, bd);
	bigint newImag = field.add(real, imag);
	newImag = field.multiply(newImag, field.add(p.real, p.imag));
	newImag = field.subtract(newImag, ac);
	newImag = field.subtract(newImag, bd);

	return Complex(field, newReal, newImag);
}

Complex Complex::divide(Complex p) {
	Complex conj = p.conjugate();
	Complex top = multiply(conj);
	bigint bottom = field.add(field.square(p.real), field.square(p.imag));
	return Complex(field, field.divide(top.real, bottom), field.divide(top.imag, bottom));
}

string Complex::str() {
	string res = real.str();
	if(imag.sign)	res += "-";
	else		res += "+";
	res += imag.str() + "i";
	return res;
}

