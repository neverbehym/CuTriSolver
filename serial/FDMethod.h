#ifndef FDMethod_h
#define FDMethod_h
#include <vector>
#include "PDE_solver.h"

using namespace std;
typedef vector<double> Vector;

class FDMethod         //finite difference method
{
   public:
    PDE_solver* pde;    //Black-scholes PDE solver
    int imax, jmax;    //0~imax: time steps; 0~jmax:asset price
    double ds, dt;
    
    vector<Vector> V;     //store option price at different time given different spot asset prices

    FDMethod(PDE_solver *pde_, int imax, int jmax);
    void ExplicitMethod();
    void ImplicitMethod();
    void CrankNicolsonMethod();
    Vector LUDecomposition(Vector q,Vector A,Vector B,Vector C);
    
    double t(double i){return dt*i;}
    double s(int j){return pde->S_min+ds*j;}
    
    double a(double i,int j){return pde->a(t(i),s(j));}
    double b(double i,int j){return pde->b(t(i),s(j));}
    double c(double i,int j){return pde->c(t(i),s(j));}
    double d(double i,int j){return pde->d(t(i),s(j));}
    

    double f (int j){return pde->f(s(j));}
    double fu(int i){return pde->fu(t(i));}
    double fl(int i){return pde->fl(t(i));}

    double getPrice(double t,double s);
};

#endif
