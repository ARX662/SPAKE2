#include <vector>

#include "pair.h"
#include "uint256_t.h"
#include "point.h"
#include "Fp.h"
#include "EllipticCurve.h"
#include "TatePairing.h"
#include "KeyGen.h"
#include "Extract.h"
#include "Encryption.h"
#include "Decryption.h"
#include "Random.h"

#define bigint uint256_t

//Inputs
//char* init_data[] = {"Hello"/*, "World"*/};
//std::vector<string> words(init_data, init_data + 1/*2*/);

static const int init_data[] = {0/*, 1, 2*/};
std::vector<int> words(init_data, init_data + sizeof(init_data)/sizeof(init_data[0]));

//All
KeyGen keys;

//Encryption
Extract extractAllTD;
Encryption enc;

std::vector<Point> sk0, sk1, sk2;
std::vector<bigint> rr;

std::vector<Complex> ciphertext;
std::vector<Point> c0, c1, c2;

//Decryption
Extract extractTDforID;
Decryption decr;

std::vector<Complex> plaintext;


void generateTD() {
	for(int i=0; i<words.size(); i++) {
		//Mapping Function
		Point h = keys.e.RandomPointInG1(Random(words[i]));
		cout << "h=" << h.str() << endl;

		// Extract
		cout << "extractAllTD" << endl;
		extractAllTD = Extract(keys.sec1, keys.e, keys.t_1, keys.t_2, keys.w, keys.g1, keys.g, h);

		sk0.push_back(extractAllTD.d_0);
		sk1.push_back(extractAllTD.d_1);
		sk2.push_back(extractAllTD.d_2);
		rr.push_back(extractAllTD.r);
	}
}

void buildEDB() {
	for(int i=0; i<words.size(); i++) {
		//Mapping Function
		Point h = keys.e.RandomPointInG1(Random(words[i]));
		cout << "h=" << h.str() << endl;

		enc = Encryption(keys.sec1, keys.e, keys.omega, keys.g1, h, keys.v_1, keys.v_2);
		ciphertext.push_back(enc.c_twil);
		c0.push_back(enc.c_0);
		c1.push_back(enc.c_1);
		c2.push_back(enc.c_2);
	}
}

void decryptEDB(int id) {
	Point h_id = keys.e.RandomPointInG1(Random(id));
	cout << "h_id=" << h_id.str() << endl;

	extractTDforID = Extract(keys.sec1, keys.e, keys.t_1, keys.t_2, keys.w, keys.g1, keys.g, h_id);
	for(int i=0; i<words.size(); i++) {
		decr = Decryption(keys.e, ciphertext.at(i), c0.at(i), c1.at(i), c2.at(i),
				 extractTDforID.d_0, extractTDforID.d_1, extractTDforID.d_2, keys.gt);
		plaintext.push_back(decr.out);
		if(decr.out.real==bigint(1)) {
			cout << "The keyword exists in the file" << endl;
			return;
		}
	}
}

int main(void) {
	cout << "GenerateTD..." << endl;
	generateTD();

	cout << "BuildEDB..." << endl;
	buildEDB();
	cout << "Data added to encrypted database." << endl;

	int search = 0;
	cout << "Searching ciphertext for keyword:'" << search << "'" << endl;	
	decryptEDB(search);
}

