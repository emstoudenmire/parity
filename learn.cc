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

template<typename SPolicy>
MPS
sampleMPS(SiteSet const& sites,
          SPolicy & sp,
          Args const& args = Args::global())
    {
    auto N = length(sites);
    auto rthreshold = args.getReal("RhoThreshold");
    auto maxdim = args.getInt("MaxDim");
    auto minsamples = args.getInt("MinSamples");
    auto nstep = args.getInt("SampleStep");
    auto pause_step = args.getBool("PauseStep",false);

    auto phi = [&sites](BitString B, int n) -> ITensor
        {
        auto p = ITensor(sites(n));
        auto elt = 1+bit(B,n);
        p.set(elt, 1.);
        return p;
        };

    auto psi = MPS(sites);

    Index eind;

    struct Sample
        {
        BitString b;
        ITensor env;
        ITensor phi;
        };
    auto computeSample = [&phi]
                         (MPS const& psi,
                          SPolicy & sp,
                          int n) -> Sample
        {
        Sample s;
        s.b = sp.next();
        s.env = ITensor(1);
        for(int k = 1; k < n; ++k)
            {
            s.env = psi(k)*s.env*phi(s.b,k);
            }
        s.phi = phi(s.b,n);
        return s;
        };

    for(int n : range1(1,N))
        {
        printfln("n = %d",n);
        sp.reset();

        ITensor rho,rho_prev;
        bool converged = false;

        auto samples = stdx::reserve_vector<Sample>(minsamples);
        for(int s = 1; s <= minsamples-nstep; ++s)
            {
            if(sp.done()) break;
            samples.push_back(computeSample(psi,sp,n));
            }

        auto trial_count = 1;
        while(not converged)
            {
            int ninclude = 0;
            rho = ITensor();
            for(int i1 = 0; i1 < samples.size(); ++i1)
            for(int i2 = i1; i2 < samples.size(); ++i2)
                {
                auto s1 = samples[i1];
                auto s2 = samples[i2];
                bool include = compFrom(n+1,s1.b,s2.b);
                if(include)
                    {
                    auto wf1 = s1.env*s1.phi;
                    auto wf2 = s2.env*s2.phi;
                    rho += wf1*prime(wf2);
                    if(i1 != i2) rho += wf2*prime(wf1);

                    ++ninclude;
                    }
                }

            if(rho)
                {
                auto TrT = rho * delta(sites(n),prime(sites(n)));
                if(eind) TrT *= delta(eind,prime(eind));
                auto Tr = elt(TrT);
                if(Tr > 0.) rho /= Tr;

                Real diff = 1.0;
                if(rho_prev)
                    {
                    diff = norm(rho-rho_prev);
                    converged = (diff < rthreshold);// && (norm(rho) > 0.);
                    }
                rho_prev = rho;

                //
                // Diagonalize rho just to visually inspect
                //
                auto [U,D,l] = diagHermitian(rho,{"Tags=","Link,"+format("n=%d",n)});
                //auto eTr = 0.;
                println("D = ");
                for(int d = 1; d <= dim(l); ++d)
                    {
                    printfln("%d  %.5f",d,elt(D,d,d));
                    //eTr += elt(D,d,d);
                    }
                //Print(eTr);
                //
                //
                //

                printfln("n=%d Trial %d: nsamples=%d,ninclude=%d, diff=%.5f",
                         n,trial_count,samples.size(),ninclude,diff);

                trial_count += 1;
                }

            //Expand the set of samples
            for(int s = 1; s <= nstep; ++s) 
                {
                if(sp.done()) break;
                samples.push_back(computeSample(psi,sp,n));
                }
            }


        auto Tr = rho * delta(sites(n),prime(sites(n)));
        if(eind) Tr *= delta(eind,prime(eind));
        rho /= elt(Tr);

        auto [U,D,l] = diagHermitian(rho,{"Tags=","Link,"+format("n=%d",n),"MaxDim=",maxdim});
        //auto [U,D,l] = diagHermitian(rho,{"Tags=","Link,"+format("n=%d",n)});
        PrintData(U);
        eind = l;
        auto eTr = 0.;
        println("D = ");
        for(int d = 1; d <= dim(l); ++d)
            {
            printfln("%d  %.5f",d,elt(D,d,d));
            eTr += elt(D,d,d);
            }
        Print(eTr);

        Real total = pow(2.,N-1);
        Real pct = samples.size()/total*100.;
        printfln("rho_%d converged after %d samples (%.3f%% of total)",n,samples.size(),pct);
        if(pause_step) PAUSE;

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
    return fabs(overlap(psi,exactMPS(sites)));
    }

struct PushSample
    {
    using storage = vector<BitString>;
    storage data;
    int n = 0;

    PushSample(storage const& d) : data(d) { }

    BitString
    next() 
        {
        auto lastn = n;
        n += 1;
        return data.at(lastn);
        }

    void
    reset() 
        {
        randshuffle(data);
        n = 0;
        }

    bool
    done() const
        {
        return (n >= data.size());
        }
    };

struct GenSample
    {
    int n = 0;
    int N = 0;

    GenSample(int N_) : N(N_) { }

    BitString
    next() 
        {
        n += 1;
        return randomEven(N);
        }

    void
    reset() 
        {
        n = 0;
        }

    bool 
    done() const { return false; }
    };

int 
main(int argc, char* argv[])
    {
    if(argc != 2) return printfln("Usage: %s inputfile",argv[0]),0;

    auto in = InputGroup(argv[1],"input");

    auto N = in.getInt("N");
    auto maxDim = in.getInt("maxDim");
    //auto ninclude = in.getInt("ninclude",0);
    auto rho_threshold = in.getReal("rho_threshold",0.1);
    auto minsamples = in.getInt("minsamples",0);
    auto samplestep = in.getInt("samplestep",1);
    auto pause_step = in.getYesNo("pause_step",false);
    auto nrepeat = in.getInt("nrepeat",1);

    auto data = allEvenStrings(N);
    printfln("Data set size = %d",data.size());
    //writeToFile(format("data%d",N),data);

    //int Nsample = 1000;
    //auto data = vector<BitString>(Nsample);
    //for(auto& s : data) s = randomEven(N);
    //println("Done making data from samples");

    auto sites = SiteSet(N,2);

    auto samplePolicy = PushSample(data);

    std::ofstream f("out.dat");

    //auto psi = makeMPS(sites,data,{"MaxDim=",maxDim});
    vector<Real> distances;
    for(int step = 1; step <= nrepeat; ++step)
        {
        auto psi = sampleMPS(sites,
                             samplePolicy,
                             {"RhoThreshold=",rho_threshold,
                              "MinSamples=",minsamples,
                              "SampleStep=",samplestep,
                              "MaxDim=",maxDim,
                              "PauseStep=",pause_step});
        auto dist = bhattDist(psi,sites);
        Print(dist);
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
        sleep(3);
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
