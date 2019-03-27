#include "itensor/all.h"
using namespace itensor;
using std::vector;
using std::string;

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

MPS
makeMPS(SiteSet const& sites,
        vector<BitString> data,
        Args const& args = Args::global())
    {
    auto N = length(sites);
    auto maxDim = args.getInt("MaxDim",100);

    auto phi = [&data,&sites](int i,int n) -> ITensor
        {
        auto s = sites(n);
        auto p = ITensor(s);
        auto elt = 1+bit(data[i],n);
        p.set(s=elt, 1.);
        return p;
        };

    int ndata = data.size();
    auto env = vector<ITensor>(ndata);
    for(auto i : range(ndata))
        {
        env[i] = phi(i,1);
        }
    auto eind = sites(1);


    auto psi = MPS(sites);

    for(int n : range1(1,N-1))
        {
        //printfln("n = %d",n);
        ITensor rho;

        for(auto i : range(ndata))
        for(int j = i; j < ndata; j += 1)
            {
            //printfln("i = %d, j = %d",i,j);
            bool include = true;
            for(int m = n+2; m <= N; m += 1)
                {
                //printfln("  n=%d bi=%d bj=%d",n,bit(data[i],n),bit(data[j],n));
                if(bit(data[i],m) != bit(data[j],m))
                    {
                    include = false;
                    break;
                    }
                }
            if(include)
                {
                auto wfi = env[i]*phi(i,n+1);
                auto wfj = env[j]*phi(j,n+1);
                rho += wfi*prime(wfj);
                if(i != j) rho += wfj*prime(wfi);
                }
            }

        auto Tr = (rho * delta(eind,prime(eind)) 
                       * delta(sites(n+1),prime(sites(n+1)))).real();
        rho /= Tr;

        ITensor U,D;
        diagHermitian(rho,U,D,{"Tags=","Link","MaxDim=",maxDim});
        //PrintData(D);

        psi.set(1+n,U);

        eind = commonIndex(U,D);
        for(auto i : range(ndata))
            {
            env[i] = U*env[i]*phi(i,n+1);
            }
        }

    ITensor U(sites(1)),D,V;
    svd(psi(2),U,D,V,{"Tags=","Link"});
    psi.set(1,U);
    psi.set(2,D*V);

    ITensor P;
    for(auto e : env) P += e;
    P /= norm(P);
    psi.set(N,psi(N)*P);

    psi.position(1);
    return psi;
    }

MPS
sampleMPS(SiteSet const& sites,
          Args const& args = Args::global())
    {
    auto N = length(sites);
    auto Nsample = args.getInt("NSample");
    auto maxdim = args.getInt("MaxDim");

    auto phi = [&sites](BitString B, int n) -> ITensor
        {
        auto p = ITensor(sites(n));
        auto elt = 1+bit(B,n);
        p.set(elt, 1.);
        return p;
        };

    auto psi = MPS(sites);

    Index eind;

    for(int n : range1(1,N))
        {
        //printfln("n = %d",n);
        ITensor rho;

        int nused = 0;

        for(auto s : range(Nsample))
            {
            auto bi = randomEven(N);
            auto bj = randomEven(N);

            //TODO: speed up by (b << n) and bi == bj
            bool include = true;
            for(int m = n+2; m <= N; m += 1)
                {
                //printfln("  n=%d bi=%d bj=%d",n,bit(data[i],n),bit(data[j],n));
                if(bit(bi,m) != bit(bj,m))
                    {
                    include = false;
                    break;
                    }
                }
            if(not include) continue;

            nused += 1;

            //printfln("bi = %d, bj = %d",bi,bj);

            //Make environment tensors
            auto envi = ITensor(1);
            auto envj = ITensor(1);
            for(int k = 1; k < n; ++k)
                {
                envi = psi(k)*envi*phi(bi,k);
                envj = psi(k)*envj*phi(bj,k);
                }

            auto wfi = envi*phi(bi,n);
            auto wfj = envj*phi(bj,n);
            rho += wfi*prime(wfj);
            if(bi == bj) rho += wfj*prime(wfi);
            }

        printfln("n=%d nused=%d",n,nused);

        auto Tr = rho * delta(sites(n),prime(sites(n)));
        if(eind) Tr *= delta(eind,prime(eind));
        rho /= elt(Tr);

        auto [U,D,l] = diagHermitian(rho,{"Tags=","Link,"+format("n=%d",n),"MaxDim=",maxdim});
        eind = l;
        Print(elt(D,1,1));
        Print(elt(D,2,2));
        println();

        psi.set(n,U);
        }

    auto P = ITensor(eind);
    for(auto s : range(Nsample))
        {
        auto bi = randomEven(N);
        auto envi = ITensor(1);
        for(int k = 1; k <= N; ++k)
            {
            envi = psi(k)*envi*phi(bi,k);
            }
        P += envi;
        }
    P /= norm(P);
    psi.set(N,psi(N)*P);

    psi.position(1);

    return psi;
    }

//returns the exact MPS for the even parity data set
MPS
exactMPS(SiteSet const& sites)
    {
    auto N = length(sites);

    auto psi = MPS(sites);

    auto link = vector<Index>(1+N);
    for(auto n : range(link)) link.at(n) = Index(2,format("Link,l=%d",n));


    for(auto n : range1(N))
        {
        auto l = link.at(n-1);
        auto r = link.at(n);
        auto s = sites(n);
        auto A = ITensor(l,s,r);

        //1 means even parity and 2 means odd

        auto val = n == 1 ? 1. : ISqrt2;

        A.set(1,1,1, val);
        A.set(2,1,2, val);

        A.set(2,2,1, val);
        A.set(1,2,2, val);

        psi.set(n, A);
        }

    auto L = ITensor(link.at(0));
    L.set(1, 1.0);
    psi.set(1, L*psi(1));

    auto R = ITensor(link.at(N));
    R.set(1, 1.0);
    psi.set(N, psi(N)*R);

    return psi;
    }

Real
bhattDist(MPS const& psi, SiteSet const& sites)
    {
    return overlap(psi,exactMPS(sites));
    }


int
main()
    {
    int N = 12;
    int Nsample = 1000;
    int maxDim = 3;

    //auto data = allEvenStrings(N);

    auto data = vector<BitString>(Nsample);
    for(auto& s : data) s = randomEven(N);
    println("Done making data");

    auto sites = SiteSet(N,2);

    //auto psi = sampleMPS(sites,{"NSample=",10000,"MaxDim=",maxDim});
    auto psi = makeMPS(sites,data,{"MaxDim=",maxDim});

    Print(bhattDist(psi,sites));

    //for(auto n : range1(20))
    //    {
    //    auto s = binarySample(psi);
    //    printf("%02d: ",n);
    //    for(auto b : s) print(b);
    //    println();
    //    }

    return 0;
    }
