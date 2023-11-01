#include <algorithm>
#include "bitstring.h"
using namespace itensor;
using std::vector;
using std::string;

template<typename Container>
void
randshuffle(Container & C)
    {
    static std::random_device rd;
    static std::mt19937 g(rd());
    std::shuffle(C.begin(), C.end(), g);
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
        printf(" %d",n);
        ITensor rho;

        for(auto i : range(ndata))
            {
            auto wfi = env[i]*phi(i,n+1);
            for(int j = i; j < ndata; j += 1)
                {
                bool include = compFrom(n+2,data[i],data[j]);
                if(include)
                    {
                    auto wfj = env[j]*phi(j,n+1);
                    rho += wfi*prime(wfj);
                    if(i != j) rho += wfj*prime(wfi);
                    }
                }
            }

        auto Tr = (rho * delta(eind,prime(eind)) 
                       * delta(sites(n+1),prime(sites(n+1)))).real();
        rho /= Tr;

        //println();
        //PrintData(rho);
        //EXIT;

        ITensor U,D;
        diagPosSemiDef(rho,U,D,{"Tags=","Link","MaxDim=",maxDim});
        //PrintData(D);

        psi.set(1+n,U);

        eind = commonIndex(U,D);
        for(auto i : range(ndata))
            {
            env[i] = U*env[i]*phi(i,n+1);
            }
        }
    println();

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


//
// Returns the exact MPS for the even parity data set
// It is a theoretical construction and known to 
// have bond dimension (tensor-train rank) equal to 2
//
MPS
exactMPS(SiteSet const& sites)
    {
    auto N = length(sites);

    auto psi = MPS(sites);

    auto link = vector<Index>(1+N);
    for(auto n : range(link)) link.at(n) = Index(2,tinyformat::format("Link,l=%d",n));


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
    return fabs(inner(psi,exactMPS(sites)));
    }

int 
main(int argc, char* argv[])
    {
    if(argc != 2) return printfln("Usage: %s inputfile",argv[0]),0;

    auto in = InputGroup(argv[1],"input");

    auto N = in.getInt("N");
    auto maxDim = in.getInt("maxDim");
    auto rho_threshold = in.getReal("rho_threshold",0.1);
    auto minsamples = in.getInt("minsamples",0);
    auto samplestep = in.getInt("samplestep",1);
    auto pause_step = in.getYesNo("pause_step",false);
    auto nrepeat = in.getInt("nrepeat",1);
    auto fraction = in.getReal("fraction",1.);
    auto amount = in.getInt("amount",0);

    println("\n");

    auto data = allEvenStrings(N);
    printfln("Data set size = %d",data.size());

    auto sites = SiteSet(N,2);

    std::ofstream f("out.dat");
    int fsize = 0;
    if(amount != 0)
        {
        fsize = amount;
        }
    else
        {
        fsize = fraction*data.size();
        }
    auto frac_data = decltype(data)(fsize);
    printfln("Actual number of training samples = %d",frac_data.size());
    printfln("Actual fraction of train samples to total data = %.5f",1.*frac_data.size()/data.size());

    vector<Real> distances;
    for(int step = 1; step <= nrepeat; ++step)
        {
        println("Shuffling data");
        randshuffle(data);
        for(auto n : range(fsize)) frac_data[n] = data[n];
    
        println("Computing MPS");
        auto psi = makeMPS(sites,
                           frac_data,
                           {"MaxDim=",maxDim});

        auto dist = bhattDist(psi,sites);
        //Print(dist);
        if(not isnan(dist)) 
            {
            distances.push_back(dist);
            printfln(f,"%.12f",dist);
            }
        println("Distances so far = ");
        Real avg = 0;
        Real avg2 = 0;
        Real Nt = distances.size();
        for(auto [n,d] : enumerate(distances))
            {
            printfln("%.12f",d);
            avg += d;
            avg2 += d*d;
            }
        avg /= Nt;
        avg2 /= Nt;
        println();
        printfln("avg = %.12f",avg);
        printfln("std dev = %.12f",sqrt(avg2-avg*avg));
        printfln("std err = %.12f",sqrt(avg2-avg*avg)/sqrt(Nt));
        }
    f.close();

    //for(auto n : range1(20))
    //    {
    //    auto s = binarySample(psi);
    //    printf("%02d: ",n);
    //    for(auto b : s) print(b);
    //    println();
    //    }

    return 0;
    }
