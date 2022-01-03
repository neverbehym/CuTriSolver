
#include "Option.h"
#include <cmath>
double N(double x)
{
    double gamma = 0.2316419;     double a1 = 0.319381530;
    double a2    =-0.356563782;   double a3 = 1.781477937;
    double a4    =-1.821255978;   double a5 = 1.330274429;
    double pi    = 4.0*atan(1.0); double k  = 1.0/(1.0+gamma*x);
    if (x>=0.0)
    {
        return 1.0-((((a5*k+a4)*k+a3)*k+a2)*k+a1)
        *k*exp(-x*x/2.0)/sqrt(2.0*pi);
    }
    else return 1.0-N(-x);
}

Option::Option(double T,double K,double sigma,double r)
{
    this->T=T;this->K=K;this->sigma=sigma;this->r=r;
}
double Option::payOff(double s) //European call option
{
    if (s>K)
        return s-K;
    return 0.0;
    
}
double Option::PriceByBSFormula(double S0)
{
    double d_plus=(log(S0/K)+(r+0.5*pow(sigma,2.0))*T)/(sigma*sqrt(T));
    double d_minus=d_plus-sigma*sqrt(T);
    return S0*N(d_plus)-K*exp(-r*T)*N(d_minus);
}
