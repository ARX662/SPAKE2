#include "Random.h"

Random::Random() {
	seed = 1;
}

Random::Random(unsigned int s) {
	seed = s;
}

unsigned int Random::nextInt() {
	seed = (8253729 * seed + 2396403);
	return seed % 32767;
}
