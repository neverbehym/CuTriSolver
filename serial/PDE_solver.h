#ifndef PDE_solver_h
#define PDE_solver_h
#include "Option.h"


class PDE_solver
{
public:
    double T,S_min,S_max;
    Option *option;
    
    PDE_solver(double S_min,double S_max,Option *option_);
    virtual double a(double t, double s)=0;
    virtual double b(double t, double s)=0;
    virtual double c(double t, double s)=0;
    virtual double d(double t, double s)=0;
    virtual double f(double s)=0;
    virtual double fu(double t)=0;
    virtual double fl(double t)=0;
    
};

class BS_PDE : public PDE_solver
{
public:

    BS_PDE(double S_min,double S_max,Option *option_):PDE_solver(S_min,S_max,option_){}
    double a(double t, double s);
    double b(double t, double s);
    double c(double t, double s);
    double d(double t, double s);
    double f(double s);
    double fl(double t);
    double fu(double t);
};

#endif
