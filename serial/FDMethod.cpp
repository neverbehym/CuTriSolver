#include "FDMethod.h"
#include <iostream>
using namespace std;
FDMethod::FDMethod(PDE_solver *pde_, int imax, int jmax)
{
   this->pde=pde_;
   this->imax=imax; this->jmax=jmax;
   this->ds=(pde->S_max - pde->S_min)/jmax;
   this->dt=pde->T/imax;
   //discretise as imax＋1 step of time-marching and jmax+1 different asset values
   V.resize(imax+1);
   for (int i=0; i<=imax; i++) V[i].resize(jmax+1);
}

void FDMethod::ExplicitMethod()
{
    for (int j=0; j<=jmax; j++) V[imax][j]=f(j);
    for (int i=imax; i>0; i--)
    {
        V[i-1][0]=fl(i-1);
        V[i-1][jmax]=fu(i-1);
        for (int j=1;j<jmax;j++)
        {
            double A=dt*(b(i,j)*0.5-a(i,j)/ds)/ds,B=1.0-dt*c(i,j)+2.0*dt*a(i,j)/(ds*ds),C=-dt*(b(i,j)*0.5+a(i,j)/ds)/ds,D=-dt*d(i,j);
            V[i-1][j]=A*V[i][j-1]+B*V[i][j]+C*V[i][j+1]+D;
        }
    }
}



void FDMethod::ImplicitMethod()
{
    Vector A(jmax),B(jmax),C(jmax),D(jmax);
    for (int j=0; j<=jmax; j++) V[imax][j]=f(j);
    for (int i=imax; i>0; i--)
    {
        for(int j=1;j<jmax;j++)
       {
        A[j]=dt*(-b(i,j)/2.0+a(i,j)/ds)/ds;
        B[j]=1.0+dt*c(i,j)-2.0*dt*a(i,j)/(ds*ds);
        C[j]=dt*(b(i,j)/2.0+a(i,j)/ds)/ds;
        D[j]=-dt*d(i,j);
        V[i][j]+=D[j];
       }
        V[i][1]+=-A[1]*fl(i-1);
        V[i][jmax-1]+=-C[jmax-1]*fu(i-1);
           
    
        V[i-1]=LUDecomposition(V[i],A,B,C);
    }
}

void FDMethod::CrankNicolsonMethod()
{
    Vector A(jmax),B(jmax),C(jmax),D(jmax),E(jmax),F(jmax),G(jmax);
    Vector q(jmax);
	for (int j = 0; j <= jmax; j++) V[imax][j] = f(j);
    for (int i=imax; i>0; i--)
    {
        for(int j=1;j<jmax;j++)
        {
            A[j]=dt*(b(i-0.5,j)*0.5-a(i-0.5,j)/ds)*0.5/ds;
            B[j]=1.0+dt*(a(i-0.5,j)/ds/ds-c(i-0.5,j)*0.5);
            C[j]=-dt*(b(i-0.5,j)*0.5+a(i-0.5,j)/ds)*0.5/ds;
            D[j]=-dt*d(i,j);
            E[j]=-A[j];
            F[j]=2-B[j];                                                                                          
            G[j]=-C[j];	
            q[j]=A[j]*V[i][j-1]+B[j]*V[i][j]+C[j]*V[i][j+1]+D[j];

        }
        q[1]+=A[1]*fl(i)-E[1]*fl(i-1);
        q[jmax-1]+=C[jmax-1]*fu(i)-G[jmax-1]*fu(i-1);
        V[i-1]=LUDecomposition(q,E,F,G);
  }    
    
    
}

//solve the tridiagonal equation by LU decomposition algorithme
//p*A=q
 Vector FDMethod::LUDecomposition(Vector q,Vector A,Vector B,Vector C)
{
    Vector p(jmax+1), r(jmax), y(jmax);
    r[1]=1/B[1];
    y[1]=q[1]*r[1];C[1]=C[1]*r[1];
    for (int j=2;j<jmax;j++)
    {
        r[j]=1/(B[j]-A[j]*C[j-1]);
        C[j]=C[j]*r[j];
        y[j]=(q[j]-A[j]*y[j-1])*r[j];
    }
    p[jmax-1]=y[jmax-1];
    for (int j=jmax-2; j>0; j--)
    {
        p[j]=y[j]-C[j]*p[j+1];
    }
    return p;
}

//return the option price at time t, given the spot price s
//interpolation
double FDMethod::getPrice(double t,double s)
{
   int i = (int)(t/dt);
   int j = (int)((s-pde->S_min)/ds);
   double l1 = (t-dt*i)/dt, l0 = 1.0-l1;
   double w1 = (s-pde->S_min-ds*j)/ds, w0 = 1.0-w1;
   return l1*w1*V[i+1][j+1] + l1*w0*V[i+1][j]+l0*w1*V[ i ][j+1] + l0*w0*V[i][j];
}
