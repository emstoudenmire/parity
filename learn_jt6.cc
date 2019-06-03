#include "itensor/all.h"
using namespace itensor;
using std::vector;
using std::string;

using BitString = unsigned int;
using Phrase = std::string;

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

int
vectparity(vector<int> v) 
    {
    int sum_of_elems = 0;
    for (auto& s : v)
    sum_of_elems += s;
    return sum_of_elems % 2;
    }

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

BitString
stringtoBitString(string s)
    {
    BitString b;
        {
        b = std::stoi( s );
        }
    return b;
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

vector<BitString> 
readData(string datafile)
    {
    std::ifstream f_in(datafile);
    if(not f_in.good()) Error("Could not find datafile / invalid file");
    vector<BitString> lines;
    printfln("Loading data from file %s...",datafile);

    for(std::string line; std::getline( f_in, line ); /**/ )
        {
        BitString b = std::stoi(line);
        lines.push_back(b);
        }
    return lines;
    }

vector<BitString>
makeDataWithReplacement(int Ntrain,int N)
    {
    auto data = vector<BitString>(Ntrain);
    for(auto& s : data)
        {s = randomEven(N);
        }
    return data;
    }   

vector<BitString> // scales exponentially! 
makeDataWithoutReplacement(int Ntrain,int N)
        {
        printfln("Making data");
        auto allbitstrings = allEvenStrings(N);
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::shuffle(allbitstrings.begin(),allbitstrings.end(),rd);
        auto data = vector<BitString>(Ntrain);
        for (int i=0; i<Ntrain; i++) 
        {
        data[i] = allbitstrings[i];
        }
        return data;
        }   
    

MPS
makeMPS(SiteSet const& sites,
        vector<BitString> data,
        Args const& args = Args::global())
    {
    auto N = length(sites);
    auto maxDim = args.getInt("MaxDim",10);
    auto verbose = args.getBool("verbose",false);
    
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
        ITensor rho;

        for(auto i : range(ndata))
        for(int j = i; j < ndata; j += 1)
            {
            // for progress monitoring with large datasets
            // if(i % 1000 == 0 and j % 1000 == 0) printfln("i = %d, j = %d",i,j);
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
                printfln("Including strings: %d=%s %d=%s",i,toString(data[i],N),j,toString(data[j],N));
                auto wfi = env[i]*phi(i,n+1);
                auto wfj = env[j]*phi(j,n+1);
                rho += wfi*prime(wfj);
                if(i != j) rho += wfj*prime(wfi);
                }
            }
        if(verbose) PrintData(rho);

        auto Tr = (rho * delta(eind,prime(eind)) 
                       * delta(sites(n+1),prime(sites(n+1)))).real();
      
        rho /= Tr;
        

        ITensor U,D;
        diagHermitian(rho,U,D,{"Tags=","Link","MaxDim=",maxDim});

        psi.set(1+n,U);

        eind = commonIndex(U,D);
        for(auto i : range(ndata))
            {
            env[i] = U*env[i]*phi(i,n+1);
            }
        }

    auto l1 = Index(dim(sites(1)),"Link");
    auto A1 = ITensor(sites(1),l1);
    for(auto n : range1(dim(sites(1))))
        {
        A1.set(n,n, 1.);
        }
    auto oldA12 = psi(2);
    psi.ref(1) = A1;
    psi.ref(2) *= delta(sites(1),l1);

    Print(norm(oldA12 - psi(1)*psi(2)));

    //ITensor U(sites(1)),D,V;
    //svd(psi(2),U,D,V,{"Tags=","Link"});
    //psi.set(1,U);
    //psi.set(2,D*V);

    ITensor P;
    for(auto e : env) P += e;
    P /= norm(P);
    psi.set(N,psi(N)*P);

    //psi.position(1);
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

        // 1 means even parity and 2 means odd

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

int
main(int argc, char* argv[])
    {
    if(argc != 2) return printfln("Usage: %s config file",argv[0]),0;
    auto in = InputGroup(argv[1],"input");
    auto N = in.getInt("N");
    auto maxDim = in.getInt("maxDim",10);
    auto fraction = in.getReal("fraction",1.); // not used
    auto Ntrain = in.getInt("Ntrain",0);
    auto outdatafile = in.getString("output_datafile","default_out.txt");
    auto load_data = in.getYesNo("load_data");
    auto with_replacement = in.getYesNo("with_replacement", true);
    auto indatafile = in.getString("input_datafile","default_in.txt");
    auto verbose = in.getYesNo("verbose", true);

    // load or make data

    std::vector<BitString> data;
    if (load_data)
        {
        data = readData(indatafile);
        int Ntrain = data.size();
        printfln("\n *** Loaded %i training examples from %d \n", Ntrain,indatafile);
        }
    else if (with_replacement)
        {
        data = makeDataWithReplacement(Ntrain,N);
        printfln("\n *** Made %i training examples with replacement \n", Ntrain);
        }
    else
        {
        data = makeDataWithoutReplacement(Ntrain,N);
        printfln("\n *** Made %i training examples without replacement \n", Ntrain);
        }


    auto sites = SiteSet(N,2);

    println("Data:");
    for(auto& d : data)
        {
        printfln("  %s",toString(d,N));
        }
    auto psi = makeMPS(sites,data,{"MaxDim=",maxDim, "verbose=", verbose});
    println("Training complete. \n");
    
    if(verbose)
        {
        println("Tensors in Psi \n");
        psi.position(N);
        for(auto i : range1(N))
            {
            PrintData(psi.A(i));
            }
        }

    float op = inner(psi,exactMPS(sites)); 
   
    printfln("Overlap with exact solution is %d \n", op);

    string outscorefile = "score_" + outdatafile;
    string outpsifile = "psi_" + outdatafile;

    printfln("Writing output files %d, %d, %d \n",outdatafile,outscorefile,outpsifile );

    std::ofstream outfile;
    outfile.open(outdatafile); 
    for(auto& s : data)
        {
            outfile<<s<<std::endl;
        }
    outfile.close();

    outfile.open(outscorefile);
    outfile<<N<<std::endl;
    outfile<<op<<std::endl;
    outfile.close();

    writeToFile(outpsifile,psi);

    return 0;
    }
