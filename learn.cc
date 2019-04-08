#include "bitstring.h"
using namespace itensor;
using std::vector;
using std::string;


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
            bool include = compFrom(n+2,data[i],data[j]);
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
    auto rthreshold = args.getReal("RhoThreshold");
    auto maxdim = args.getInt("MaxDim");
    auto minsamples = args.getInt("MinSamples");
    auto nstep = 64;

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
        printfln("n = %d",n);
        ITensor rho,rho_prev;
        bool converged = false;

        auto samples = stdx::reserve_vector<BitString>(minsamples);
        for(int s = 1; s <= minsamples-nstep; ++s)
            {
            samples.push_back(randomEven(N));
            }

        while(not converged)
            {
            rho = ITensor();
            for(int i1 = 0; i1 < samples.size(); ++i1)
            for(int i2 = i1; i2 < samples.size(); ++i2)
                {
                auto b1 = samples[i1];
                auto b2 = samples[i2];
                bool include = compFrom(n+1,b1,b2);
                if(include)
                    {
                    //Make environment tensors
                    auto env1 = ITensor(1);
                    auto env2 = ITensor(1);
                    for(int k = 1; k < n; ++k)
                        {
                        env1 = psi(k)*env1*phi(b1,k);
                        env2 = psi(k)*env2*phi(b2,k);
                        }

                    auto wf1 = env1*phi(b1,n);
                    auto wf2 = env2*phi(b2,n);
                    rho += wf1*prime(wf2);
                    if(i1 != i2) rho += wf2*prime(wf1);
                    }
                }

            auto Tr = rho * delta(sites(n),prime(sites(n)));
            if(eind) Tr *= delta(eind,prime(eind));
            rho /= elt(Tr);

            if(rho_prev)
                {
                converged = (norm(rho-rho_prev) < rthreshold);
                }
            rho_prev = rho;

            for(int s = 1; s <= nstep; ++s) samples.push_back(randomEven(N));
            }

        printfln("rho_%d converged after %d samples",n,samples.size());

        auto Tr = rho * delta(sites(n),prime(sites(n)));
        if(eind) Tr *= delta(eind,prime(eind));
        Print(Tr);
        rho /= elt(Tr);
        PrintData(rho);

        auto [U,D,l] = diagHermitian(rho,{"Tags=","Link,"+format("n=%d",n),"MaxDim=",maxdim});
        //auto [U,D,l] = diagHermitian(rho,{"Tags=","Link,"+format("n=%d",n)});
        PrintData(U);
        Print(norm(rho-U*D*prime(U)));
        eind = l;
        auto eTr = 0.;
        println("D = ");
        for(int d = 1; d <= dim(l); ++d)
            {
            printfln("%d  %.5f",d,elt(D,d,d));
            eTr += elt(D,d,d);
            }
        Print(eTr);
        //PAUSE;

        psi.set(n,U);
        }

    auto P = ITensor(eind);
    ITensor Pprev;
    bool converged = false;
    int step = 0;
    while(not converged)
        {
        auto bi = randomEven(N);
        auto envi = ITensor(1);
        for(int k = 1; k <= N; ++k)
            {
            envi = psi(k)*envi*phi(bi,k);
            }
        P += envi;
        P /= norm(P);
        step += 1;
        if(Pprev)
            {
            converged = (norm(P-Pprev) < rthreshold);
            }
        Pprev = P;
        if(step > 10000) break;
        }
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
    int N = 6;
    int maxDim = 3;

    //auto data = allEvenStrings(N);

    //int Nsample = 1000;
    //auto data = vector<BitString>(Nsample);
    //for(auto& s : data) s = randomEven(N);
    //println("Done making data from samples");

    auto sites = SiteSet(N,2);

    //auto psi = makeMPS(sites,data,{"MaxDim=",maxDim});
    auto psi = sampleMPS(sites,{"RhoThreshold=",1E-2,
                                "MinSamples=",500,
                                "MaxDim=",maxDim});

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
