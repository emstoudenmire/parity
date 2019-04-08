#include "bitstring.h"
using namespace itensor;
using std::vector;
using std::string;


int main()
    {
    auto N = 8;
    auto b1 = randomEven(N);
    auto b2 = randomEven(N);

    for(int n = 0; n <= N; ++n)
        {
        Print(n);
        Print(toString(b1 >> n,N));
        }

    Print(compFrom(2,b1,b2));
    Print(compFrom(2,b1,b1));
    Print(compFrom(2,b2,b2));

    Print(toString(64,8));
    Print(toString(64+3,8));
    Print(toString(64+128,8));
    Print(compFrom(3,64,64+3));
    Print(compFrom(3,64,64+128));
    for(int n = 1; n <= N; ++n)
        {
        printfln("bit(64+3,%d) = %d",n,bit(64+3,n));
        }

    return 0;
    }
