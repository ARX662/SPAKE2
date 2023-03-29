#include "pair.h"
#include "uint256_t.h"
#define bigint uint256_t

#include "point.h"
#include "Fp.h"
#include "EllipticCurve.h"
#include "TatePairing.h"
#include "KeyGen.h"

int main(void) {
	// Check maximum value
	cout << endl;
	bigint max = 1;
	max <<= 255;
	max = max + (max - bigint(1));
	cout << "max 256 bit value = " << max.str() << endl;
	cout << endl;

	// This is a 127 bit prime
	//"122365946062993565078966846847292526143"
	uint128_t low(6633471227987084073, 5293657587167514175);
	uint128_t up(0, 0);
	bigint p(up, low);
	cout << "Modular ring is created by " << p.bits() << " bits prime number:" << endl;
	cout << "p = " << p.str() << endl;

	// Create modular arithmetic domain using prime p
        Fp fp(p);

	cout << endl;

	// ======= Test numbers ========
	cout << "Some test numbers:" << endl;

  	//29041337592471582783003652909236324819	
	uint128_t lowax(1574334065482131953, 11310291793859938771);
	bigint ax(up, lowax);
	cout << "ax = " << ax.str() << endl;

	//22669445037153944759606560802932763159
	uint128_t loway(1228913077916152184, 8079615676273633815);
	bigint ay(up, loway);
	cout << "ay = " << ay.str() << endl;


	//87586307157543603111551745087418933304
	uint128_t lowbx(4748063225009573058, 1454045674644971576);
	bigint bx(up, lowbx);
	cout << "bx = " << bx.str() << endl;

	//39404405148284318446227366519574278943
	uint128_t lowby(2136117083363442677, 10936625631785562911);
	bigint by(up, lowby);
	cout << "by = " << by.str() << endl;

        cout << endl;

	cout << "================================================================================" << endl;

	// ======= Test modular arithmetic ring ========	
	cout << "Testing modular ring:" << endl;

	bigint c = fp.add(ax, bx);
	cout << "ax + bx: " << c.str() << endl;

	bigint d = fp.add(ax, p);
	cout << "ax + p: " << d.str() << endl;

	cout << endl;

	d = fp.multiply(ax, ax);
	cout << "ax x ax: " << d.str() << endl;

	cout << endl;

	// modular multiplication inverse (i) of integer (n) is
	// defined as: n x i = 1 
	bigint inv = fp.inverse(ax); 
	cout << "inverse of ax: " << inv.str() << endl;

	cout << endl;
	cout << "================================================================================" << endl;

	// ========== Elliptic Curve ===============
	
	EllipticCurve ec(&fp);
	ec.a4 = 1;
	cout << "Elliptic curve: y^2 = x^3 + x  (using above modular ring)" << endl;

	cout << endl;

	// Number pairs to be used in Elliptic curve calculations
	Point p1(ax, ay);
	Point p2(bx, by);
	cout << "Point p1: " << p1.x.str() << " " << p1.y.str() << endl;
	cout << "Point p2: " << p2.x.str() << " " << p2.y.str() << endl;

	// This gives wrong result because of the problem with modular multiplication (fp.multiply)
	// If Fp::multiply is used instead of Fp::multiply in ec.multiply definition it gives
	// correct result
	Point p3 = ec.multiply(p1, p2.x);
	cout << "p1 x p2 : " << p3.x.str() << " " << p3.y.str() << endl;

	return 0;
}
