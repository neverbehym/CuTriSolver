\*The code is adapted on 'Core numerical methods in finance with c, Cambridge Press'*\



#include "Option.h"
#include "PDE_solver.h"
#include "FDMethod.h"
#include <iostream>
#include <chrono>

using namespace std;

int main()
{
    chrono::time_point<std::chrono::system_clock> start, end1,end2,end3;
    double S0=103.0, r=0.05, sigma=0.2;
    double T=1.0/12, K=100.0, s_min=0.0, s_max=200.0;
    
    Option option(T,K,sigma,r);
    cout<<"formula price"<<option.PriceByBSFormula(S0)<<endl;
 
   
 PDE_solver *p=new BS_PDE(s_min,s_max,&option);
     int imax=3000, jmax=300;
   FDMethod example1(p,imax,jmax); 
    start = chrono::system_clock::now();
    example1.ExplicitMethod();
    end1 = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds = end1-start;
    cout << "explicit method: Price = " << example1.getPrice(0.0,S0) << endl;
   cout<<"time consuming: "<<elapsed_seconds.count()<<endl;

    imax=200, jmax=2000;
    FDMethod example2(p,imax,jmax);
    start = chrono::system_clock::now();
    example2.ImplicitMethod();
    end2 = chrono::system_clock::now();
    elapsed_seconds = end2-start; 
    cout << "implicit method: Price = " << example2.getPrice(0.0,S0) << endl;
    cout<<"time consuming: "<<elapsed_seconds.count()<<endl;
    
    
    imax=200, jmax=2000;
    FDMethod example3(p,imax,jmax);
    start = chrono::system_clock::now();
    example3.CrankNicolsonMethod();
    end3 = chrono::system_clock::now();
    elapsed_seconds = end3-start;
    
    cout << "Crank Nicolson method: Price = " << example3.getPrice(0.0,S0) << endl;
    cout<<"time consuming: "<<elapsed_seconds.count()<<endl;
    

    
    return 0;
}
