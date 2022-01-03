
#define TIMESTEPS 200
#define ITERATION 100
#define SYSTEMS 1024
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <helper_cuda.h>

#include <iostream>

// the function to compute reciprocal operation is  provided by Mike Giles ,2014    
//****************************************************************************
static __forceinline__ __device__ float __rcp(float a) {
  return 1.0f / a;
}

static __forceinline__ __device__ double __rcp(double a) {

#if __CUDA_ARCH__ >= 300  //if the compute capability >3.0
  double e, y; 
  //The approximate reciprocal function in the Special Dunction Unit
  // y=1/a
  asm ("rcp.approx.ftz.f64 %0, %1;" : "=d"(y) : "d"(a)); 
  // __fma_rn(x,y,z)  x*y+z
  // refined to full double precision by extension to Newton iteration
  // a mathmatical method to reduce the relative error
  e = __fma_rn (-a, y, 1.0); //-a*y+1.0
  e = __fma_rn ( e, e,   e); //-e*e+e
  y = __fma_rn ( e, y,   y); //-e*y+y
  return y; 
#else
  return 1.0 / a;
#endif
}
//*****************************************************************************



////////////////////////////////////////////////////////////////////
//                                                                //
//                    Thomas algorithm                   (serial) //
//                                                                //
////////////////////////////////////////////////////////////////////

template <typename REAL>
void Thomas(REAL *a,REAL *b,REAL *c,REAL *d,int size)
{
//forward:tansform the tridiagonal matrix to upper diagonal matrix
REAL c_new[size];
    c_new[0]=c[0]/b[0];d[0]=d[0]/b[0];
    for (int i=1;i<size;i++)
    {
	REAL r=1/(b[i]-a[i]*c_new[i-1]);
        c_new[i]=c[i]*r;
        d[i]=(d[i]-a[i]*d[i-1])*r;
    }
	
//backward substitution, get d[size-1] and then backward
    for (int i=size-2; i>=0; i--)
    {
        d[i]=d[i]-c_new[i]*d[i+1];
    }

}

/*This kernel is adapted on 'Fast tridiagonal solvers on GPU' by Zhang, Largely modified*/
////////////////////////////////////////////////////////////////////
//                                                                //
//                        Cyclic reduction                        //
//                                                                //
////////////////////////////////////////////////////////////////////
template <typename REAL>
__global__ void crKernel(REAL *d_a, REAL *d_b, REAL *d_c, REAL *d_d)
{
	int t_id = threadIdx.x;
	int b_id = blockIdx.x;
 //tridiagonal system size is twice of blockDim.x, but still ,solving one system with one block  
    	int size = blockDim.x * 2;   
    	int iteration = (int)log2(REAL(size))-1;   //  log2(size)-1

    	extern __shared__ char shared[];


    REAL* a = (REAL*)shared;
    REAL* b = (REAL*)&a[size];
    REAL* c = (REAL*)&b[size];
    REAL* d = (REAL*)&c[size];
    REAL* x = (REAL*)&d[size];

//read global menory into shared memory
    a[t_id] = d_a[t_id];
    a[t_id + blockDim.x] = d_a[t_id + blockDim.x]; //a[0:2*blockDim.x-1]

    b[t_id] = d_b[t_id ];
    b[t_id + blockDim.x] = d_b[t_id + blockDim.x]; //b[0:2*blockDim.x-1]

    c[t_id] = d_c[t_id+b_id*2*blockDim.x];
    c[t_id + blockDim.x] = d_c[t_id + blockDim.x+b_id*2*blockDim.x]; //c[0:2*blockDim.x-1]

    d[t_id] = d_d[t_id ];
    d[t_id + blockDim.x] = d_d[t_id + blockDim.x ]; //d[0:2*blockDim.x-1]

   // It is essential to set a[0] = c[size-1] = 0	
    if(t_id==0) a[0]=0.0;
    if(t_id==blockDim.x-1) c[size-1]=0.0;
    int stride = 1;
    int size_reduced=size/2;

    //forward elimination
//only perform the transformatin on even equations of every new reduced tridiagonal system
//Index (1,3,5,7...size-1) then Index (3,7,11...size-1) then Index (7,15,23...size-1)...at last,Index (size/2-1,size-1)
    for (int j = 0; j <iteration; j++)
    {
        __syncthreads();
	if (threadIdx.x < size_reduced)
	{
	        /*get the even-indexed i of the reduced system
		  stride is the distance from i to the upper or lower equation
                  on which we perform the linear combination*/
		int i = stride * 2*t_id + stride*2 - 1;
		int up = i - stride;
		int down = min(i + stride,size-1);

		REAL r_up = a[i] *__rcp(b[up]);
		REAL r_down = c[i] *__rcp(b[down]);
		b[i] = b[i] - c[up] * r_up - a[down] * r_down;
		d[i] = d[i] - d[up] * r_up - d[down] * r_down;
		a[i] = -a[up] * r_up;
		c[i] = -c[down] * r_down;   

 //__syncthreads() not required as even equations is updating, whose data depend only on odd coefficients


	}
        stride *= 2;
        size_reduced /= 2;
      
    }


    // now stride=size/2
    // solve the final reduced tridiagonal system: num1=size/2-1,num2=size-1
    // twin euqations (1) b[i]*x[i]+c[i]*x[j]=d[i]
    //                (2) a[j]*x[i]+b[j]*x[j]=d[j]
    if (t_id ==0)
    {
      int i = stride - 1;
      int j = 2 * stride - 1;
      REAL tmp = b[j]*b[i]-c[i]*a[j];
      x[i] = (b[j]*d[i]-c[i]*d[j])*__rcp(tmp);
      x[j] = (d[j]*b[i]-d[i]*a[j])*__rcp(tmp);
    }

    // backward substitution
    size_reduced = 2;
    for (int k = 0; k <iteration; k++)
    {
        __syncthreads();
	stride /= 2;
        if (t_id < size_reduced)
        { 
//only substitute the solved ones in odd equations of every orignal tridiagonal system that were not transformed
//Index (size/4-1,size*3/4-1) then Index (size/8-1,size*3/8-1,size*5/8-1,size*7/8-1)...at last,Index (0,2,4,6..size-2)
            int i = stride * 2*t_id + stride - 1;
            if(i == stride - 1)   //first point in the odd reduced tridiagonal system in every step
                  x[i] = (d[i] - c[i]*x[i+stride])*__rcp(b[i]);
            else
                  x[i] = (d[i] - a[i]*x[i-stride] - c[i]*x[i+stride])*__rcp(b[i]);
         }
	
         size_reduced *= 2;
      }

    __syncthreads();

    d_d[t_id+b_id*2*blockDim.x] = x[t_id];
    d_d[t_id + blockDim.x+b_id*2*blockDim.x] = x[t_id + blockDim.x];
}

////////////////////////////////////////////////////////////////////
//                                                                //
//            Cyclic reduction  (No bank conflict)                //
//                                                                //
////////////////////////////////////////////////////////////////////

// memory padding, redirect the index and pad one unit every 32 units
// suit for modern GPU with 32 banks
__forceinline__ __device__ int pad(int x)
{
    int warpid = x / 32;
    int laneid = x % 32; 
//shift forward one index every warp 
    int y = warpid * 33 + laneid;
    return y;
}

// cyclic reduction
template <typename REAL>
__global__ void crNBCKernel(REAL *d_a, REAL *d_b, REAL *d_c, REAL *d_d)
{
    int t_id = threadIdx.x;	
    int b_id = blockIdx.x;
    int size = blockDim.x * 2;
    int iteration = (int)log2(REAL(size))-1;
  
//the number of spaces of each fragment =the number of warps
    int ext_size = size * 33 / 32;
    extern __shared__ char shared[];
    REAL* a = (REAL*)shared;
    REAL* b = (REAL*)&a[ext_size];
    REAL* c = (REAL*)&b[ext_size];
    REAL* d = (REAL*)&c[ext_size];
    REAL* x = (REAL*)&d[ext_size];

//shift forward one index every warp when reading from global memory 
    a[pad(t_id)] =d_a[t_id+b_id*2*blockDim.x];
    a[pad(t_id + blockDim.x)] = d_a[t_id + blockDim.x+b_id*2*blockDim.x];

    b[pad(t_id)] = d_b[t_id+b_id*2*blockDim.x];
    b[pad(t_id + blockDim.x)] = d_b[t_id + blockDim.x+b_id*2*blockDim.x];

    c[pad(t_id)] = d_c[t_id+b_id*2*blockDim.x];
    c[pad(t_id + blockDim.x)] = d_c[t_id + blockDim.x+b_id*2*blockDim.x];

    d[pad(t_id)] = d_d[t_id+b_id*2*blockDim.x];
    d[pad(t_id + blockDim.x)] = d_d[t_id + blockDim.x+b_id*2*blockDim.x];
   // It is essential to set a[0] = c[size-1] = 0	
     if(t_id==0) a[0]=0.0;
     if(t_id==blockDim.x-1) c[pad(size-1)]=0.0;


    int stride = 1;
    int size_reduced=size/2;
    //forward elimination
//only perform the transformatin on even equations of every new reduced tridiagonal system
//Index (1,3,5,7...size-1) then Index (3,7,11...size-1) then Index (7,15,23...size-1)...at last,Index (size/2-1,size-1)
    for (int k = 0; k <iteration; k++)
    {
   __syncthreads();

        if (t_id < size_reduced)
	{ 

		int i = stride * 2*t_id + stride*2 - 1;
		int up = max(i - stride,0);
		int down = min(i + stride,size-1);


                //redirect to the shifted index
		i = pad(i);
		up = pad(up);
		down = pad(down);

		REAL r_up = a[i]*__rcp(b[up]);
		REAL r_down = c[i] *__rcp(b[down]);
		b[i] = b[i] - c[up] * r_up - a[down] * r_down;
		d[i] = d[i] - d[up] * r_up - d[down] * r_down;
		a[i] = -a[up] * r_up;
		c[i] = -c[down] * r_down;
                
	}
        size_reduced /= 2;
	stride*=2;
    }

    // now stride=size/2
    // solve the final reduced tridiagonal system: i=size/2-1,j=size-1
    // twin euqations (1) b[i]*x[i]+c[i]*x[j]=d[i]
    //                (2) a[j]*x[i]+b[j]*x[j]=d[j]
     if (t_id ==0)
    {
      int i = stride - 1;
      int j = 2 * stride - 1;
      i=pad(i);
      j=pad(j);
      REAL tmp = b[j]*b[i]-c[i]*a[j];
      x[i] = (b[j]*d[i]-c[i]*d[j])*__rcp(tmp);
      x[j] = (d[j]*b[i]-d[i]*a[j])*__rcp(tmp);
    }

    // backward substitution
//only substitute the solved ones in odd equations of every orignal tridiagonal system that were not transformed
//Index (size/4-1,size*3/4-1) then Index (size/8-1,size*3/8-1,size*5/8-1,size*7/8-1)...at last,Index (0,2,4,6..size-2)
    stride=size/2;
    size_reduced = 2;
    for (int j = 0; j <iteration; j++)
    {
        __syncthreads();
        stride /= 2;
        if (t_id < size_reduced)
        {
            
            int i = stride * 2*t_id + stride - 1;
	    int up = pad(i - stride);
	    int down = pad(i + stride);
            if(i == stride - 1)   
                  x[pad(i)] = (d[pad(i)] - c[pad(i)]*x[down]) *__rcp(b[pad(i)]);
            else
                  x[pad(i)] = (d[pad(i)] - a[pad(i)]*x[up] - c[pad(i)]*x[down])*__rcp( b[pad(i)]);
         }
         size_reduced *= 2;
      }

    __syncthreads();

    d_d[t_id +b_id*2*blockDim.x] = x[pad(t_id)];
    d_d[t_id + blockDim.x +b_id*2*blockDim.x] = x[pad(t_id + blockDim.x)];
}



////////////////////////////////////////////////////////////////////
//                                                                //
//            Cyclic reduction  (split into two blocks)           //
//                                                                //
////////////////////////////////////////////////////////////////////

// 
template <typename REAL>
__global__ void crNBCX2Kernel(REAL *d_a, REAL *d_b, REAL *d_c, REAL *d_d,REAL *d_y,REAL *d_z,REAL *d_alpha)
{
    int t_id = threadIdx.x;	
    int b_id = blockIdx.x;
    int index=t_id+b_id*blockDim.x;
    int iteration = (int)log2(REAL(blockDim.x))-1;
  
//the number of spaces of each fragment =the number of warps
    int ext_size = blockDim.x * 33 / 32;
    extern __shared__ char shared[];
    REAL* a = (REAL*)shared;
    REAL* b = (REAL*)&a[ext_size];
    REAL* c = (REAL*)&b[ext_size];
    REAL* d = (REAL*)&c[ext_size];
    REAL* e = (REAL*)&d[ext_size];

//shift forward one index every warp when reading from global memory 
    a[pad(t_id)] =d_a[index];
    b[pad(t_id)] = d_b[index];
    c[pad(t_id)] = d_c[index];
    d[pad(t_id)] = d_d[index];
    e[pad(t_id)] = 0.0;

   // It is essential to set a[0] = c[size-1] = 0	
     if(t_id==0) {a[0]=0.0; if(b_id%2==1) e[0]=1.0;}
     if(t_id==blockDim.x-1) {c[pad(blockDim.x-1)]=0.0;if(b_id%2==0) e[blockDim.x-1]=1.0;}

    int stride = 1;
    int size_reduced=blockDim.x/2;
    //forward elimination
//only perform the transformatin on even equations of every new reduced tridiagonal system
//Index (1,3,5,7...size-1) then Index (3,7,11...size-1) then Index (7,15,23...size-1)...at last,Index (size/2-1,size-1)
    for (int k = 0; k <iteration; k++)
    {
        __syncthreads();

        if (t_id < size_reduced)
	{ 

		int i = stride * 2*t_id + stride*2 - 1;
		int up = max(i - stride,0);
		int down = min(i + stride,blockDim.x-1);


                //redirect to the shifted index
		i = pad(i);
		up = pad(up);
		down = pad(down);

		REAL r_up = a[i]*__rcp(b[up]);
		REAL r_down = c[i] *__rcp(b[down]);
		b[i] = b[i] - c[up] * r_up - a[down] * r_down;
		d[i] = d[i] - d[up] * r_up - d[down] * r_down;
		e[i] = e[i] - e[up] * r_up - e[down] * r_down;
		a[i] = -a[up] * r_up;
		c[i] = -c[down] * r_down;
                
	}
        size_reduced /= 2;
	stride*=2;
    }

    // now stride=size/2
    // solve the final reduced tridiagonal system: i=size/2-1,j=size-1
    // twin euqations (1) b[i]*x[i]+c[i]*x[j]=d[i]
    //                (2) a[j]*x[i]+b[j]*x[j]=d[j]
     if (t_id ==0)
    {
      int i = stride - 1;
      int j = 2 * stride - 1;
      int ii=pad(i);
      int jj=pad(j);
      REAL tmp = b[jj]*b[ii]-c[ii]*a[jj];
      d_y[b_id*blockDim.x+i] = (b[jj]*d[ii]-c[ii]*d[jj])*__rcp(tmp);
      d_y[b_id*blockDim.x+j] = (d[jj]*b[ii]-d[ii]*a[jj])*__rcp(tmp);
      d_z[b_id*blockDim.x+i] = (b[jj]*e[ii]-c[ii]*e[jj])*__rcp(tmp);
      d_z[b_id*blockDim.x+j] = (e[jj]*b[ii]-e[ii]*a[jj])*__rcp(tmp);

    }

    // backward substitution
//only substitute the solved ones in odd equations of every orignal tridiagonal system that were not transformed
//Index (size/4-1,size*3/4-1) then Index (size/8-1,size*3/8-1,size*5/8-1,size*7/8-1)...at last,Index (0,2,4,6..size-2)
    stride=blockDim.x/2;
    size_reduced = 2;
    for (int j = 0; j <iteration; j++)
    {
        __syncthreads();
        stride /= 2;
        if (t_id < size_reduced)
        {
            
            int i = stride * 2*t_id + stride - 1;
	    int up =i - stride;
	    int down = i + stride;
            if(i == stride - 1){   
                  d_y[b_id*blockDim.x+i] = (d[pad(i)] - c[pad(i)]*d_y[b_id*blockDim.x+down]) *__rcp(b[pad(i)]);
                  d_z[b_id*blockDim.x+i] = (e[pad(i)] - c[pad(i)]*d_z[b_id*blockDim.x+down]) *__rcp(b[pad(i)]);
		}
            else{
                  d_y[b_id*blockDim.x+i]= (d[pad(i)] - a[pad(i)]*d_y[b_id*blockDim.x+up] - c[pad(i)]*d_y[b_id*blockDim.x+down])*__rcp( b[pad(i)]);
                  d_z[b_id*blockDim.x+i]= (e[pad(i)] - a[pad(i)]*d_z[b_id*blockDim.x+up] - c[pad(i)]*d_z[b_id*blockDim.x+down])*__rcp( b[pad(i)]);
		}
         }
         size_reduced *= 2;
        __syncthreads();
      }

//      (z[blockDim.x-1]    1/d_a[blockDim&  &])    (alpha1)   (-d_y[[blockDim.x-1])     b[i] c[i] d[i]
//solve                                      *           =  
//      (1/d_c(blockDim-1]   z[blockDim.x] )    (alpha2)   (-d_y[[blockDim.x])       a[j] b[j] d[j]
    if(t_id==0 && b_id==0){
	REAL tmp=  d_z[blockDim.x+b_id*blockDim.x] *d_z[blockDim.x-1+b_id*blockDim.x]-__rcp(d_a[blockDim.x+b_id*blockDim.x]*d_c[blockDim.x-1+b_id*blockDim.x]);
	d_alpha[b_id]= (-d_d[blockDim.x-1+b_id*blockDim.x]*d_z[blockDim.x+b_id*blockDim.x]+d_y[blockDim.x+b_id*blockDim.x]*__rcp(d_a[blockDim.x+b_id*blockDim.x]))*__rcp(tmp);
	d_alpha[b_id+1] = (-d_d[blockDim.x+b_id*blockDim.x]*d_z[blockDim.x-1+b_id*blockDim.x]+d_y[blockDim.x-1+b_id*blockDim.x]*__rcp(d_c[blockDim.x-1+b_id*blockDim.x]) )*__rcp(tmp);
    }
   

//    (y1)  (alpha1*z1)
// x=     + 
//    (y2)  (alpha2*z2)
    __syncthreads();
    d_d[index]=d_y[index]+d_alpha[b_id]*d_z[index];
}

/*This kernel is adapted on 'Fast tridiagonal solvers on GPU' by Zhang, Largely modified*/
////////////////////////////////////////////////////////////////////
//                                                                //
//                    Parallel cyclic reduction                   //
//                                                                //
////////////////////////////////////////////////////////////////////
template <typename REAL>
__global__ void pcrKernel(REAL *d_a, REAL *d_b, REAL *d_c, REAL *d_d)
{
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int size=blockDim.x;

//solve one system with one block
    extern __shared__ char shared[];

    //a,b,c,d point the address of corresponding fragment in shared memory array
    REAL* a = (REAL*)shared;
    REAL* b = (REAL*)&a[size+1]; 
    REAL* c = (REAL*)&b[size+1];  
    REAL* d = (REAL*)&c[size+1];
 
   //read all the coefficients of each system
    a[t_id] = d_a[t_id+b_id*size];
    b[t_id] = d_b[t_id+b_id*size];
    c[t_id] = d_c[t_id+b_id*size];
    d[t_id] = d_d[t_id+b_id*size];
   // It is essential to set a[0] = c[size-1] = 0	
    if(t_id==0) a[0]=0.0;
    if(t_id==size-1) c[size-1]=0.0;
   __syncthreads();

    //parallel cyclic reduction for every points
    //start with stride=1,get the size/2 pairs of equations after log2(size)-1
    int iteration = (int)log2(REAL(size))-1;
    int stride = 1;
    for (int k = 0; k <iteration; k++)
    {
        
        REAL r_up=0.0;REAL r_down=0.0;

        int down = t_id+stride;
	if (down >= size) down = size-1;
	else  r_down = c[t_id]*__rcp(b[down]); //r_down*row[down]

        int up = t_id-stride;
        if (up < 0) up = 0;
	else r_up = a[t_id] *__rcp(b[up]) ;     //r_up*row[up]
       
	__syncthreads();  
        
        //new system
	REAL b_new = b[t_id]-c[up]*r_up-a[down]*r_down;  
        REAL d_new = d[t_id]-d[down]*r_down-d[up]*r_up;
	REAL a_new = -a[up]*r_up;
	REAL c_new = -c[down]*r_down;
	__syncthreads();  
        a[t_id]=a_new;b[t_id]=b_new;c[t_id]=c_new;d[t_id]=d_new;
	 stride *=2; 
	 __syncthreads(); 
	
    }
	
    // solve the system of twins equations. now stride=size/2
    // b[i]x[i]+c[i]x[j]=d[i]
    // a[j]x[i]+b[j]x[j]=d[j]
    	REAL x;
    if (t_id < stride)
    {
		
        int i = t_id;
        int j = t_id+stride;
        REAL tmp = b[j]*b[i]-c[i]*a[j];
        x = (b[j]*d[i]-c[i]*d[j])*__rcp(tmp);
    }
    if (t_id >= stride)
    {
		
        int i = t_id-stride;
        int j = t_id;
        REAL tmp = b[j]*b[i]-c[i]*a[j];
        x = (d[j]*b[i]-d[i]*a[j])*__rcp(tmp);
    }
    __syncthreads();
   d_d[t_id+b_id*size] = x;

}


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//        Parallel cyclic reduction (split into two blocks)             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
template <typename REAL>
__global__ void pcrX2Kernel(REAL *d_a, REAL *d_b, REAL *d_c, REAL *d_d,REAL *d_z,REAL *d_alpha)
{
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int index=b_id*blockDim.x+t_id;

   //split the system into two blocks/sections
    extern __shared__ char shared[];

    //a,b,c,d,e,y,z point the address of corresponding fragment in shared memory array
    REAL* a = (REAL*)shared;
    REAL* b = (REAL*)&a[blockDim.x+1]; 
    REAL* c = (REAL*)&b[blockDim.x+1];  
    REAL* d = (REAL*)&c[blockDim.x+1]; 
    REAL* e = (REAL*)&d[blockDim.x+1];
 

   //read all the coefficients (for block 0 and block 1)
    a[t_id] = d_a[index];
    b[t_id] = d_b[index];
    c[t_id] = d_c[index];
    d[t_id] = d_d[index];
    e[t_id]=0.0; 
    __syncthreads();

   // It is essential to set a[0] = c[blockDim.x-1] = 0	
    if(t_id==0){
	a[0]=0.0;
	if(b_id%2==0) e[blockDim.x-1]=1.0;
	else e[0]=1.0;
    }
    if(t_id==blockDim.x-1) c[blockDim.x-1]=0.0;
    
    __syncthreads();
    
    //slove the tridiagonal systems REAL_i*y_i=d_i REAL_i*z_i=e_i for i=0,1
    //parallel cyclic reduction for every points
    //start with stride=1,get the blockDim.x/2 pairs of equations after log2(blockDim.x)-1
    int iteration = (int)log2(REAL(blockDim.x))-1;
    int stride = 1;
    for (int k = 0; k <iteration; k++)
    {
        
        REAL r_up=0.0;REAL r_down=0.0;

        int down = t_id+stride;
	if (down >= blockDim.x) down = blockDim.x-1;
	else  r_down = c[t_id] *__rcp(b[down]); //r_down*row[down]

        int up = t_id-stride;
        if (up < 0) up = 0;
	else r_up = a[t_id] *__rcp(b[up]);     //r_up*row[up]
       

        //new system
	REAL b_new = b[t_id]-c[up]*r_up-a[down]*r_down;  
	REAL a_new = -a[up]*r_up;
	REAL c_new = -c[down]*r_down;
        REAL d_new = d[t_id]-d[down]*r_down-d[up]*r_up;
        REAL e_new = e[t_id]-e[down]*r_down-e[up]*r_up;

	__syncthreads();  
        a[t_id]=a_new;b[t_id]=b_new;c[t_id]=c_new;d[t_id]=d_new;e[t_id]=e_new;
	stride *=2; 
	 __syncthreads(); 
	
    }
	
    // solve the system of twins equations. now stride=size/2
    // b[i]x[i]+c[i]x[j]=d[i]
    // a[j]x[i]+b[j]x[j]=d[j]
     REAL y,z;
    if (t_id < stride)
    {
		
        int i = t_id;
        int j = t_id+stride;
        REAL tmp = b[j]*b[i]-c[i]*a[j];
        y = (b[j]*d[i]-c[i]*d[j])*__rcp(tmp);
        z = (b[j]*e[i]-c[i]*e[j])*__rcp(tmp);
     }
    if (t_id >= stride)    
    {
		
        int i = t_id-stride;
        int j = t_id;
        REAL tmp = b[j]*b[i]-c[i]*a[j];
        y = (d[j]*b[i]-d[i]*a[j])*__rcp(tmp);
        z = (d[j]*b[i]-e[i]*a[j])*__rcp(tmp);
     }
     __syncthreads();
     d_d[index] = y;
     if(t_id==blockDim.x-1 && b_id%2==0)     d_z[b_id] = z;
     if(t_id==0 && b_id%2==1) d_z[b_id] = z;

     __syncthreads();


//      (z[blockDim.x-1]    1/d_a[blockDim])    (alpha1)   (-d_d[[blockDim.x-1])     b[i] c[i] d[i]
//solve                                      *           =  
//      (1/d_c(blockDim-1]   z[blockDim.x] )    (alpha2)   (-d_d[[blockDim.x])       a[j] b[j] d[j]

    if(t_id==0 && b_id%2==0){
	REAL tmp=  d_z[b_id+1] *d_z[b_id]-__rcp(d_a[b_id*blockDim.x+blockDim.x]*d_c[b_id*blockDim.x+blockDim.x-1]);
	d_alpha[b_id]= (-d_d[b_id*blockDim.x+blockDim.x-1]*d_z[b_id+1]+d_d[b_id*blockDim.x+blockDim.x]*__rcp(d_a[b_id*blockDim.x+blockDim.x]))*__rcp(tmp);
	d_alpha[b_id+1] = (-d_d[b_id*blockDim.x+blockDim.x]*d_z[b_id]+d_d[b_id*blockDim.x+blockDim.x-1]*__rcp(d_c[b_id*blockDim.x+blockDim.x-1]) )*__rcp(tmp);
    }
   

//    (y1)  (alpha1*z1)
// x=     + 
//    (y2)  (alpha2*z2)
    __syncthreads();
    d_d[index]=y+d_alpha[b_id]*z;
}



////////////////////////////////////////////////////////////////////
//                                                                //
//        Parallel cyclic reduction + cyclic reduction            //
//                                                                //
////////////////////////////////////////////////////////////////////
template <typename REAL>
__global__ void pcr_cr_Kernel(REAL *d_a, REAL *d_b, REAL *d_c, REAL *d_d)
{
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
// one block per system, size=2*blocksize
// forward elimination once by CR
// solve the reduced system by PCR, size=blocksize
// backward substitute by CR to get the other half unknowns


// store the odd and even coefficients and right hands seperately
    extern __shared__ char shared[];
    REAL* a1 = (REAL*)shared;                // index:0,2,4,6 ....2*blocksize-2
    REAL* a2 = (REAL*)&a1[blockDim.x+1];     //index: 1,3,5,7....2*blocksize-1
    REAL* b1 = (REAL*)&a2[blockDim.x+1];
    REAL* b2 = (REAL*)&b1[blockDim.x+1];
    REAL* c1 = (REAL*)&b2[blockDim.x+1];
    REAL* c2 = (REAL*)&c1[blockDim.x+1];
    REAL* d1 = (REAL*)&c2[blockDim.x+1];
    REAL* d2 = (REAL*)&d1[blockDim.x+1];

//a1,b1,c1 don't change over iteration
    a1[t_id] = d_a[t_id*2+b_id*2*blockDim.x];  
    b1[t_id] = d_b[t_id*2+b_id*2*blockDim.x];
    c1[t_id] = d_c[t_id*2+b_id*2*blockDim.x];    
    
    d1[t_id] = d_d[t_id*2+b_id*2*blockDim.x];
    d2[t_id] = d_d[t_id*2+1+b_id*2*blockDim.x];
    
    if(t_id==0) a1[t_id]=0.0;

    for(int k=0;k<ITERATION;k++){

     a2[t_id] = d_a[t_id*2+1+b_id*2*blockDim.x];
     b2[t_id] = d_b[t_id*2+1+b_id*2*blockDim.x];
     c2[t_id] = d_c[t_id*2+1+b_id*2*blockDim.x];
     if(t_id==blockDim.x-1) c2[t_id]=0.0;
    __syncthreads();

    
// step 1: forward elimination by CR, only once

        // reduced to half size of the system,
	//a1[i]*x[2i-1]+b1[i]*x[2i]+c1[i]*x[2i+1]=d1[i]
        //a2[i]*x[2i]+b2[i]*x[2i+1]+c2[i]*x[2i+2]=d2[i]   <<<<only update the even equations 
        //a1[i+1]*x[2i+1]+b1[i+1]*x[2i+2]+c1[i+1]*x[2i+3]=d1[i+1]

	

	REAL r_up = a2[t_id] *__rcp( b1[t_id]);
	REAL r_down =0.0;
	if (t_id == blockDim.x-1){
		b2[t_id] = b2[t_id] - c1[t_id] * r_up  ;
		d2[t_id] = d2[t_id] - d1[t_id] * r_up  ;
		a2[t_id] = -a1[t_id] * r_up;
	}
	else{	
 		r_down= c2[t_id] *__rcp( b1[t_id+1]);
		b2[t_id] = b2[t_id] - c1[t_id] * r_up  - a1[t_id+1] * r_down ;
		d2[t_id] = d2[t_id] - d1[t_id] * r_up  - d1[t_id+1] * r_down ;
		a2[t_id] = -a1[t_id] * r_up;
		c2[t_id] = -c1[t_id+1] * r_down ;
	}
        __syncthreads();    

// step 2: solve the half even-indexed system by PCR
    int stride = 1;
    int iteration = (int)log2(REAL(blockDim.x))-1;
    //parallel cyclic reduction
    for (int i = 0; i <iteration; i++)
    {

	REAL r_up=0.0;REAL r_down=0.0;
	int down = t_id+stride;
	if (down >= blockDim.x) down = blockDim.x-1;
	else  r_down = c2[t_id]*__rcp(b2[down]); //r_down*row[down]

        int up = t_id-stride;
        if (up < 0) up = 0;
	else r_up = a2[t_id] *__rcp(b2[up]) ;     //r_up*row[up]
       
	__syncthreads();    

	REAL bNew = b2[t_id] - c2[up] * r_up - a2[down] * r_down;
	REAL dNew = d2[t_id] - d2[up] * r_up - d2[down] * r_down;
	REAL aNew = -a2[up] * r_up;
	REAL cNew = -c2[down] * r_down;
   
	__syncthreads();
	a2[t_id] = aNew;b2[t_id] = bNew;c2[t_id] = cNew;d2[t_id] = dNew;
        __syncthreads();
        stride*=2;
		
    }
    //stride=blocksize/2
    	REAL x;
    if (t_id < stride)
    {
		
        int i = t_id;
        int j = t_id+stride;
        REAL tmp = b2[j]*b2[i]-c2[i]*a2[j];
        x = (b2[j]*d2[i]-c2[i]*d2[j])*__rcp(tmp);
    }
    if (t_id >= stride)
    {
		
        int i = t_id-stride;
        int j = t_id;
        REAL tmp = b2[j]*b2[i]-c2[i]*a2[j];
        x = (d2[j]*b2[i]-d2[i]*a2[j])*__rcp(tmp);
    }

    d2[t_id]=x;

        __syncthreads();
// step 3: backward substitution in CR

    if(t_id == 0)
     	d1[t_id] = (d1[t_id] - c1[t_id]*d2[t_id])*__rcp(b1[t_id]);
    else
        d1[t_id] = (d1[t_id] - a1[t_id]*d2[t_id-1] - c1[t_id]*d2[t_id])*__rcp(b1[t_id]);
    __syncthreads();
    
    }

    d_d[2*t_id+b_id*2*blockDim.x] = d1[t_id];
    d_d[2*t_id+1+b_id*2*blockDim.x] = d2[t_id];
}


////////////// for the double precision of 2048 size system only/////////////
// for the double precision of 2048 size system only

template <typename REAL>
__global__ void pcr_cr_LKernel(REAL *d_a, REAL *d_b, REAL *d_c, REAL *d_d)
{
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
// one block per system, size=2*blocksize
// forward elimination once by CR
// solve the reduced system by PCR, size=blocksize
// backward substitute by CR to get the other half unknowns


// store the odd and even coefficients and right hands seperately
    extern __shared__ char shared[];

    REAL* a2 = (REAL*)shared;     //index: 1,3,5,7....2*blocksize-1
    REAL* b2 = (REAL*)&a2[blockDim.x+1];
    REAL* c2 = (REAL*)&b2[blockDim.x+1];
    REAL* d1 = (REAL*)&c2[blockDim.x+1];
    REAL* d2 = (REAL*)&d1[blockDim.x+1];
 
    
    d1[t_id] = d_d[t_id*2+b_id*2*blockDim.x];
    d2[t_id] = d_d[t_id*2+1+b_id*2*blockDim.x];
    
    if(t_id==0) d_a[b_id*2*blockDim.x]=0.0;
    __syncthreads();
    for(int k=0;k<ITERATION;k++){

     a2[t_id] = d_a[t_id*2+1+b_id*2*blockDim.x];
     b2[t_id] = d_b[t_id*2+1+b_id*2*blockDim.x];
     c2[t_id] = d_c[t_id*2+1+b_id*2*blockDim.x];
     if(t_id==blockDim.x-1) c2[t_id]=0.0;
    __syncthreads();

    
// step 1: forward elimination by CR, only once

        // reduced to half size of the system,
	//d_a[2i]*x[2i-1]+d_b[2i]*x[2i]+d_c[2i]*x[2i+1]=d1[i]
        //a2[i]*x[2i]+b2[i]*x[2i+1]+c2[i]*x[2i+2]=d2[i]   <<<<only update the even equations 
        //d_a[2i+2]*x[2i+1]+d_b[2i+2]*x[2i+2]+d_c[2i+2]*x[2i+3]=d1[i+1]

	
	REAL r_up = a2[t_id] *__rcp( d_b[2*t_id+b_id*2*blockDim.x]);
	REAL r_down =0.0;
	if (t_id == blockDim.x-1){
		b2[t_id] = b2[t_id] - d_c[2*t_id+b_id*2*blockDim.x] * r_up  ;
		d2[t_id] = d2[t_id] - d1[t_id] * r_up  ;
		a2[t_id] = -d_a[2*t_id+b_id*2*blockDim.x] * r_up;
	}
	else{	
 		r_down= c2[t_id]  *__rcp(d_b[2*t_id+2+b_id*2*blockDim.x]);
		b2[t_id] = b2[t_id] - d_c[2*t_id+b_id*2*blockDim.x] * r_up  - d_a[2*t_id+2+b_id*2*blockDim.x] * r_down ;
		d2[t_id] = d2[t_id] - d1[t_id] * r_up  - d1[t_id+1] * r_down ;
		a2[t_id] = -d_a[2*t_id+b_id*2*blockDim.x] * r_up;
		c2[t_id] = -d_c[2*t_id+2+b_id*2*blockDim.x] * r_down ;
	}
        __syncthreads();    

// step 2: solve the half even-indexed system by PCR
    int stride = 1;
    int iteration = (int)log2(REAL(blockDim.x))-1;
    //parallel cyclic reduction
    for (int i = 0; i <iteration; i++)
    {

	REAL r_up=0.0;REAL r_down=0.0;
	int down = t_id+stride;
	if (down >= blockDim.x) down = blockDim.x-1;
	else  r_down = c2[t_id]*__rcp(b2[down]); //r_down*row[down]

        int up = t_id-stride;
        if (up < 0) up = 0;
	else r_up = a2[t_id] *__rcp(b2[up]) ;     //r_up*row[up]
       
	__syncthreads();    

	REAL bNew = b2[t_id] - c2[up] * r_up - a2[down] * r_down;
	REAL dNew = d2[t_id] - d2[up] * r_up - d2[down] * r_down;
	REAL aNew = -a2[up] * r_up;
	REAL cNew = -c2[down] * r_down;
   
	__syncthreads();
	a2[t_id] = aNew;b2[t_id] = bNew;c2[t_id] = cNew;d2[t_id] = dNew;
        __syncthreads();
        stride*=2;
		
    }
    //stride=blocksize/2
    	REAL x;
    if (t_id < stride)
    {
		
        int i = t_id;
        int j = t_id+stride;
        REAL tmp = b2[j]*b2[i]-c2[i]*a2[j];
        x = (b2[j]*d2[i]-c2[i]*d2[j])*__rcp(tmp);
    }
    if (t_id >= stride)
    {
		
        int i = t_id-stride;
        int j = t_id;
        REAL tmp = b2[j]*b2[i]-c2[i]*a2[j];
        x = (d2[j]*b2[i]-d2[i]*a2[j])*__rcp(tmp);
    }

    d2[t_id]=x;

        __syncthreads();
// step 3: backward substitution in CR

    if(t_id == 0)
     	d1[t_id] = (d1[t_id] - d_c[2*t_id+b_id*2*blockDim.x]*d2[t_id])*__rcp(d_b[2*t_id+b_id*2*blockDim.x]);
    else
        d1[t_id] = (d1[t_id] - d_a[2*t_id+b_id*2*blockDim.x]*d2[t_id-1] - d_c[2*t_id+b_id*2*blockDim.x]*d2[t_id])*__rcp(d_b[2*t_id+b_id*2*blockDim.x]);
    __syncthreads();
    
    }

    d_d[2*t_id+b_id*2*blockDim.x] = d1[t_id];
    d_d[2*t_id+1+b_id*2*blockDim.x] = d2[t_id];
}


// below code from this point to end of this file is borrowed from Mike Giles
// with some parts modified 


////////////////////////////////////////////////////////////////////
//          Parallel cyclic reduction within a warp               //

// one row per thread
template <typename REAL>
__forceinline__ __device__ 
void PCRwarp(REAL a, REAL b, REAL c, REAL &y)
{ 
    int stride=1;
    if(threadIdx.x%32==0) a=0.0;
    if(threadIdx.x%32==31) c=0.0;
#pragma unroll
    for (int n=0; n<5; n++) {
    	REAL r_down = c*__rcp(__shfl_down(b,stride));
  	REAL r_up = a *__rcp(__shfl_up(b,stride));
  	b= b-__shfl_up(c,stride)*r_up-__shfl_down(a,stride)*r_down;  
    	a=-__shfl_up(a,stride)*r_up;
    	c=-__shfl_down(c,stride)*r_down;  
    	y= y-__shfl_down(y,stride)*r_down-__shfl_up(y,stride)*r_up;
   	 stride = stride<<1;
    }
	y= y*__rcp(b);
}


// 2 rows per thread.
// every thread holds am,cm,dm,ap,cp,dp
// system size=64 
// solve A*x=d 
//
//     (  1.0  cm[0]                               )      ( dm[0]  )
//     ( ap[0]  1.0  cp[0]                         )      ( dp[0]  )
//     (       am[1]  1.0  cm[1]                   )      ( dm[1]  )
//     (             ap[1]  1.0  cp[1]             )      ( dp[1]  )
// A = (               .     .    .                ), d = (        )
//     (                     .    .    .           )      (   .    )
//     (                                           )      (   .    )
//     (                         am[31] 1.0  cm[31])      ( dm[31] )
//     (                               ap[31] 1.0  )      ( dp[31] )
//

// 
// This code uses optimisations due to Jeremy Appleyard (NVIDIA)

template <typename REAL>
__forceinline__ __device__ 
void PCR_2_warp(REAL am, REAL cm, REAL &dm,REAL ap, REAL cp, REAL &dp){
  
  REAL r;
  int stride=1;
if(threadIdx.x%32==0) am=0.0;
if(threadIdx.x%32==31) cp=0.0;
  r   = __rcp(1.0 - ap*cm - cp*__shfl_down(am,1));
  dp  =  (dp  - ap*dm - cp*__shfl_down(dm,1))*r; 
  ap  = - ap*am*r;
  cp  = - cp*__shfl_down(cm,1)*r; 

#pragma unroll
  for (int n=0; n<5; n++) {
    r   = __rcp(1.0 - ap*__shfl_up(cp,stride) - cp*__shfl_down(ap,stride));
    dp  =   r*(dp - ap*__shfl_up(dp,stride) - cp*__shfl_down(dp,stride));
    ap  = - r*ap*__shfl_up(ap,stride);
    cp  = - r*cp*__shfl_down(cp,stride);

    stride *= 2;
  }
 //updated dm,dp are the every two rows of the solutions for Ax=d, 
  dm = dm - am*__shfl_up(dp,1) - cm*dp; 
}


////////////////////////////////////////////////////////////////////
//                                                                //
//              Parallel cyclic reduction +Thomas                 //
//                                                                //
////////////////////////////////////////////////////////////////////


// solve a tridiagonal system within the warp,with 'subsize' rows per thread
// where subsize=systemsize/32
// perform modefied Thomas algorithm within the thread, perform PCR in warp scale


//system size=512
template <typename REAL>
__launch_bounds__(256, 6) //MAX_THREADS_PER_BLOCK ,MIN_BLOCKS_PER_MP 
// use  40 registers per thread
__global__ void pcr_Thomas_warp1(REAL *d_a, REAL *d_b, REAL *d_c, REAL *d_d){

    
    int t_id=threadIdx.x;
    int b_id=blockIdx.x;
// read coefficients into local variables
    int subsize=16;
    REAL a[16], b[16],c[16], d[16];
    for(int i=0;i<subsize;i++){
	int  index = (t_id+b_id*blockDim.x)*subsize+i;
	a[i]=d_a[index];b[i]=d_b[index];c[i]=d_c[index];d[i]=d_d[index];
    }

    // It is essential for the user to set a[0] of the first thread = c[31] of the last thread = 0
    if (t_id%32==0) a[0]=0.0;
    if (t_id%32==31) c[subsize-1]=0.0;

    REAL r;
    REAL atem[16],  ctem[16], dtem[16];
// step 1:Use modified Thomas algorithm to obtain equation system being expressed in terms of end values,
// i.e a[i]*x[0]+x[i]+c[i]*x[SIZE-1]=d[i]
   for (int j=0;j<ITERATION;j++){
    //forward
    	for (int i=0; i<2; i++) {
   		r  = __rcp(b[i]);
    		dtem[i] = r * d[i];
    		atem[i] = r * a[i];
   		ctem[i] = r * c[i];
   	 }
	
    	for (int i=2; i<subsize; i++) {
    		r  =   __rcp( b[i] - a[i]*ctem[i-1] );
    		dtem[i] =  r * ( d[i] - a[i]*dtem[i-1] );
    		atem[i] =  r * (      - a[i]*atem[i-1] );
   		ctem[i] =  r *   c[i];
    	}

    //backward
    	for (int i=subsize-3; i>0; i--) {
    		dtem[i] =  dtem[i] - ctem[i]*dtem[i+1];
    		atem[i] =  atem[i] - ctem[i]*atem[i+1];
   		ctem[i] =       - ctem[i]*ctem[i+1];
    	}

    	r  = __rcp( 1.0f - ctem[0]*atem[1] );
   	 dtem[0] =  r * ( dtem[0] - ctem[0]*dtem[1] );
   	 atem[0] =  r *   atem[0];
   	 ctem[0] =  r * (      - ctem[0]*ctem[1] );


//step 2: Use cyclic reduction algorithm to get end values d[0],d[SIZE-1] of each thread(section)
    	PCR_2_warp(atem[0],ctem[0],dtem[0],atem[subsize-1],ctem[subsize-1],dtem[subsize-1]);

//step 3: solve all other central values by substituting end values in the equation system acquired in step 1
    	d[0]=dtem[0]; d[subsize-1]=dtem[subsize-1];
//d_d[t_id*subsize]=d[0];d_d[t_id*subsize+subsize-1]=d[subsize-1];
    	for (int i=1; i<subsize-1; i++) {dtem[i]=dtem[i] - atem[i]*dtem[0] - ctem[i]*dtem[subsize-1];d[i]=dtem[i];}
		

    }
	for(int i=0;i<subsize;i++) {int  index = (t_id+b_id*blockDim.x)*subsize+i; d_d[index]= d[i];}
}


//system size=1024
template <typename REAL>
__launch_bounds__(256, 4) //MAX_THREADS_PER_BLOCK ,MIN_BLOCKS_PER_MP 
// use  64 registers per thread
__global__ void pcr_Thomas_warp2(REAL *d_a, REAL *d_b, REAL *d_c, REAL *d_d){

    
    int t_id=threadIdx.x;
    int b_id=blockIdx.x;
// read coefficients into local variables
    int subsize=32;
    REAL a[32], b[32],c[32], d[32];
    for(int i=0;i<subsize;i++){
	int  index = (t_id+b_id*blockDim.x)*subsize+i;
	a[i]=d_a[index];b[i]=d_b[index];c[i]=d_c[index];d[i]=d_d[index];
    }

    // It is essential for the user to set a[0] of the first thread = c[31] of the last thread = 0
    if (t_id%32==0) a[0]=0.0;
    if (t_id%32==31) c[subsize-1]=0.0;

    REAL r;
    REAL atem[32],  ctem[32], dtem[32];
// step 1:Use modified Thomas algorithm to obtain equation system being expressed in terms of end values,
// i.e a[i]*x[0]+x[i]+c[i]*x[SIZE-1]=d[i]
   for (int j=0;j<ITERATION;j++){
    //forward
    	for (int i=0; i<2; i++) {
   		r  = __rcp(b[i]);
    		dtem[i] = r * d[i];
    		atem[i] = r * a[i];
   		ctem[i] = r * c[i];
   	 }
	
    	for (int i=2; i<subsize; i++) {
    		r  =   __rcp( b[i] - a[i]*ctem[i-1] );
    		dtem[i] =  r * ( d[i] - a[i]*dtem[i-1] );
    		atem[i] =  r * (      - a[i]*atem[i-1] );
   		ctem[i] =  r *   c[i];
    	}

    //backward
    	for (int i=subsize-3; i>0; i--) {
    		dtem[i] =  dtem[i] - ctem[i]*dtem[i+1];
    		atem[i] =  atem[i] - ctem[i]*atem[i+1];
   		ctem[i] =       - ctem[i]*ctem[i+1];
    	}

    	r  = __rcp( 1.0f - ctem[0]*atem[1] );
   	 dtem[0] =  r * ( dtem[0] - ctem[0]*dtem[1] );
   	 atem[0] =  r *   atem[0];
   	 ctem[0] =  r * (      - ctem[0]*ctem[1] );


//step 2: Use cyclic reduction algorithm to get end values d[0],d[SIZE-1] of each thread(section)
    	PCR_2_warp(atem[0],ctem[0],dtem[0],atem[subsize-1],ctem[subsize-1],dtem[subsize-1]);

//step 3: solve all other central values by substituting end values in the equation system acquired in step 1
    	d[0]=dtem[0]; d[subsize-1]=dtem[subsize-1];
//d_d[t_id*subsize]=d[0];d_d[t_id*subsize+subsize-1]=d[subsize-1];
    	for (int i=1; i<subsize-1; i++) {dtem[i]=dtem[i] - atem[i]*dtem[0] - ctem[i]*dtem[subsize-1];d[i]=dtem[i];}
		

    }
	for(int i=0;i<subsize;i++) {int  index = (t_id+b_id*blockDim.x)*subsize+i; d_d[index]= d[i];}
}

//system size=2048
template <typename REAL>
__launch_bounds__(256,2) //MAX_THREADS_PER_BLOCK ,MIN_BLOCKS_PER_MP 
// use  128 registers per thread
__global__ void pcr_Thomas_warp3(REAL *d_a, REAL *d_b, REAL *d_c, REAL *d_d){

    
    int t_id=threadIdx.x;
    int b_id=blockIdx.x;
// read coefficients into local variables
    int subsize=64;
    REAL a[64], b[64],c[64], d[64];
    for(int i=0;i<subsize;i++){
	int  index = (t_id+b_id*blockDim.x)*subsize+i;
	a[i]=d_a[index];b[i]=d_b[index];c[i]=d_c[index];d[i]=d_d[index];
    }

    // It is essential for the user to set a[0] of the first thread = c[31] of the last thread = 0
    if (t_id%32==0) a[0]=0.0;
    if (t_id%32==31) c[subsize-1]=0.0;

    REAL r;
    REAL atem[64],  ctem[64], dtem[64];
// step 1:Use modified Thomas algorithm to obtain equation system being expressed in terms of end values,
// i.e a[i]*x[0]+x[i]+c[i]*x[SIZE-1]=d[i]
   for (int j=0;j<ITERATION;j++){
    //forward
    	for (int i=0; i<2; i++) {
   		r  = __rcp(b[i]);
    		dtem[i] = r * d[i];
    		atem[i] = r * a[i];
   		ctem[i] = r * c[i];
   	 }
	
    	for (int i=2; i<subsize; i++) {
    		r  =   __rcp( b[i] - a[i]*ctem[i-1] );
    		dtem[i] =  r * ( d[i] - a[i]*dtem[i-1] );
    		atem[i] =  r * (      - a[i]*atem[i-1] );
   		ctem[i] =  r *   c[i];
    	}

    //backward
    	for (int i=subsize-3; i>0; i--) {
    		dtem[i] =  dtem[i] - ctem[i]*dtem[i+1];
    		atem[i] =  atem[i] - ctem[i]*atem[i+1];
   		ctem[i] =       - ctem[i]*ctem[i+1];
    	}

    	r  = __rcp( 1.0f - ctem[0]*atem[1] );
   	 dtem[0] =  r * ( dtem[0] - ctem[0]*dtem[1] );
   	 atem[0] =  r *   atem[0];
   	 ctem[0] =  r * (      - ctem[0]*ctem[1] );


//step 2: Use cyclic reduction algorithm to get end values d[0],d[SIZE-1] of each thread(section)
    	PCR_2_warp(atem[0],ctem[0],dtem[0],atem[subsize-1],ctem[subsize-1],dtem[subsize-1]);

//step 3: solve all other central values by substituting end values in the equation system acquired in step 1
    	d[0]=dtem[0]; d[subsize-1]=dtem[subsize-1];
//d_d[t_id*subsize]=d[0];d_d[t_id*subsize+subsize-1]=d[subsize-1];
    	for (int i=1; i<subsize-1; i++) {dtem[i]=dtem[i] - atem[i]*dtem[0] - ctem[i]*dtem[subsize-1];d[i]=dtem[i];}
		

    }
	for(int i=0;i<subsize;i++) {int  index = (t_id+b_id*blockDim.x)*subsize+i; d_d[index]= d[i];}
}






















