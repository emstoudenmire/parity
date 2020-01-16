#include "bitstring.h"
using namespace itensor;
using std::vector;
using std::string;

int
main()
    {
    int N = 8;
    int Nsample = 1000;
    int maxDim = 3;

    auto data = allEvenStrings(N);

    auto sites = SiteSet(N,2);

    return 0;
    }
