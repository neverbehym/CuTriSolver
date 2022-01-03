#include "PDE_solver.h"
#include <iostream>
#include <cmath>
PDE_solver::PDE_solver(double S_min,double S_max,Option *option_)
{
    T=option_->T;
    this->S_max=S_max;this->S_min=S_min;
    this->option=option_;
}

double BS_PDE::a(double t, double s)
{
   return -0.5*pow(option->sigma*s,2);
}

double BS_PDE::b(double t, double s)
{
   return -option->r*s;
}

double BS_PDE::c(double t, double s)
{
   return option->r;
}
double BS_PDE::d(double t, double s)
{
    return 0.0;
}




double BS_PDE::f(double s)
{
   return option->payOff(s);
    
}

double BS_PDE::fl(double t)
{
    return 0.0;
}

double BS_PDE::fu(double t)
{
    return option->payOff(S_max)*exp(-option->r*(T-t));
}

