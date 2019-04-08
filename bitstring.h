#include "itensor/all.h"

using std::vector;
using std::string;
using std::array;

namespace itensor {

using BitString = unsigned int;

bool
randBool()
    {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::bernoulli_distribution d(0.5);
    return d(gen);
    }

BitString
randomEven(int N)
    {
    BitString res = 0;
    int par = 1;
    unsigned int place = 1;
    for(int i = 1; i <= N-1; i += 1)
        {
        auto bit = randBool();
        par *= (1-2*bit);
        res += bit*place;
        place *= 2;
        }
    auto bit = (1-par)/2;
    res += bit*place;
    return res;
    }

//1-indexed
int
bit(BitString b, int n)
    {
    auto s = b >> (n-1);
    return s%2;
    }

//0 means even parity, 1 means odd
int
parity(BitString b) { return __builtin_parity(b); }

string
toString(BitString b, int N)
    {
    string s;
    for(auto n : range1(N))
        {
        s += bit(b,n)==0 ? '0' : '1';
        }
    return s;
    }

//scales exponentially!
vector<BitString>
allStrings(int N)
    {
    auto res = vector<BitString>(pow(2,N));
    for(auto n : range(res)) res[n] = n;
    return res;
    }

//scales exponentially!
vector<BitString>
allEvenStrings(int N)
    {
    auto all = allStrings(N);;
    auto even = vector<BitString>(all.size()/2);
    auto n = 0;
    for(auto b : all) if(parity(b) == 0)
        {
        even[n] = b;
        n += 1;
        }
    return even;
    }

//sample a single bit string from a normalized MPS
vector<int>
binarySample(MPS W)
    {
    auto N = length(W);
    auto state = vector<int>(1+N,0);

    for(auto j : range1(N))
        {
        auto s = siteIndex(W,j);
        auto Pup = setElt(s(1),prime(s)(1));

        auto pup = (dag(prime(W(j),"Site"))*Pup*W(j)).real();
        auto pdn = 1.0-pup;

        if(pup > 1.000001)
            {
            printfln("pup = %.2E",pup);
            Error("pup > 1");
            }

        auto r = Global::random();

        auto& st = state.at(j);
        st = (r < pup) ? 0 : 1;

        auto proj = setElt(s(1+st));

        if(j < N)
            {
            auto p = (st==0) ? pup : pdn;
            
            W.Aref(j+1) *= (proj * W(j));
            W.Aref(j+1) /= std::sqrt(p);
            }
        }
    return state;
    }

//
// Compare the bit strings b1 and b2
// starting with the n'th bit
// Returns true if bits n,n+1,...,N match
// 
bool inline
compFrom(int n,
         BitString b1,
         BitString b2)
    {
    auto shift = n-1;
    return (b1 >> shift) == (b2 >> shift);
    }

} //namespace itensor
