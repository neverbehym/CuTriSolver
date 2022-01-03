#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cuda.h>
#include <curand.h>
#include <helper_cuda.h>
#include "tridiagonal_solver.h"
#include <iostream>
#include <fstream>



///////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                               //
// implicit scheme of BS model with PCR+Thoms algorithm by shuffle to solve tridiagonal systems  //
//                                                                  			         //
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename REAL>
__launch_bounds__(256,4) 
__global__ void implicit_pcrThomas(REAL sigma,REAL Smax,REAL K,REAL T,REAL r,REAL *d_x,REAL *d_upperbound){


    int t_id=threadIdx.x;
    int b_id=blockIdx.x;

// set coefficients into local variables
    // num_points=num_threads+2 (512+2) subsize=512/32
	// (including the two end points defined by boundary function)
	int subsize=16;
	REAL deltaS = Smax/ 513;   
	REAL deltaT = T/  TIMESTEPS;
	
    REAL a[16], b[16],c[16], d[16],cmax;
    for(int i=0;i<subsize;i++){
	//S from 1* deltaS to blockDim.x* deltaS (excluding Smin=0, Smax=(blockDim.x+1)* deltaS )
	REAL S = (t_id%32*subsize+i+1)* deltaS;
	REAL alpha = 0.5*deltaT*sigma*sigma *S*S/(deltaS*deltaS);
	REAL beta = 0.5*deltaT*r*S/deltaS;
	a[i] = -alpha + beta;
	b[i] = 1.0f + 2*alpha +r*deltaT;
	c[i] = -alpha - beta;
	if(i==15) cmax=c[i];   // store c[size-1] in the last row
	//initialise the payoff as the right hand of tridiagonal system d_d
	//vanilla European call option	
	d[i] = max(S-K,0.0);
    }


    // It is essential for the user to set a[0] of the first thread = c[31] of the last thread = 0
    if (t_id%32==0) a[0]=0.0;
    if (t_id%32==31) c[subsize-1]=0.0;

    REAL ratio;
    REAL atem[16],  ctem[16], dtem[16];
// step 1:Use modified Thomas algorithm to obtain equation system being expressed in terms of end values,
// i.e a[i]*x[0]+x[i]+c[i]*x[SIZE-1]=d[i]
   for (int j=0;j<TIMESTEPS;j++){
    //forward

   // perform the transformation to the last row, related to the upper bound at the time rightnow  
     if(t_id%32==31) d[subsize-1]+= -cmax*d_upperbound[j+1];

    	for (int i=0; i<2; i++) {
   		ratio  = __rcp(b[i]);
    		dtem[i] = ratio * d[i];
    		atem[i] = ratio * a[i];
   		ctem[i] = ratio * c[i];
   	 }
	
    	for (int i=2; i<subsize; i++) {
    		ratio  =   __rcp( b[i] - a[i]*ctem[i-1] );
    		dtem[i] =  ratio * ( d[i] - a[i]*dtem[i-1] );
    		atem[i] =  ratio * (      - a[i]*atem[i-1] );
   		    ctem[i] =  ratio *   c[i];
    	}

    //backward
    	for (int i=subsize-3; i>0; i--) {
    		dtem[i] =  dtem[i] - ctem[i]*dtem[i+1];
    		atem[i] =  atem[i] - ctem[i]*atem[i+1];
   			ctem[i] =       - ctem[i]*ctem[i+1];
    	}

    	ratio  = __rcp( 1.0f - ctem[0]*atem[1] );
   	 	dtem[0] =  ratio * ( dtem[0] - ctem[0]*dtem[1] );
   	 	atem[0] =  ratio *   atem[0];
   	 	ctem[0] =  ratio * (      - ctem[0]*ctem[1] );


//step 2: Use cyclic reduction algorithm to get end values d[0],d[SIZE-1] of each thread(section)
    	PCR_2_warp(atem[0],ctem[0],dtem[0],atem[subsize-1],ctem[subsize-1],dtem[subsize-1]);

//step 3: solve all other central values by substituting end values in the equation system acquired in step 1
    	d[0]=dtem[0]; d[subsize-1]=dtem[subsize-1];
    	for (int i=1; i<subsize-1; i++) {dtem[i]=dtem[i] - atem[i]*dtem[0] - ctem[i]*dtem[subsize-1];d[i]=dtem[i];}
		

    }
	for(int i=0;i<subsize;i++) {int  index = (t_id+b_id*blockDim.x)*subsize+i; d_x[index]= d[i];}
}




/////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                         //
//    CN scheme of BS model with CR+PCR algorithm of solving tridiagonal systems     //
//                                                                  			   //
/////////////////////////////////////////////////////////////////////////////////////////////
template <typename REAL>
__global__  void CN_crpcr(REAL sigma,REAL Smax,REAL K,REAL T,REAL r,REAL *d_a1,REAL *d_b1,REAL *d_c1,REAL *d_x,REAL* d_upperbound)
{
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;


// set the odd (d_a1[0:blocksize*blocknum-1]) and even coefficients (a2[0:blocksize-1])and right hands (d1,d2) seperately
    extern __shared__ char shared[];

    REAL* a2 = (REAL*)shared;     //index: 1,3,5,7....2*blocksize-1
    REAL* b2 = (REAL*)&a2[blockDim.x+1];
    REAL* c2 = (REAL*)&b2[blockDim.x+1];
    REAL* d1 = (REAL*)&c2[blockDim.x+1];
    REAL* d2 = (REAL*)&d1[blockDim.x+1];
 
    // num_points=num_threads+2 (including the two end points defined by boundary function)
	REAL deltaS = Smax/ (2*blockDim.x+1);   
	REAL deltaT = T/  TIMESTEPS;
   //S from 1* deltaS to blockDim.x* deltaS (excluding Smin=0, Smax=(blockDim.x+1)* deltaS )
	REAL S1 = (t_id*2+1)* deltaS;
	REAL S2 = (t_id*2+2)* deltaS;
	REAL alpha1 = 0.5*deltaT*sigma*sigma *S1*S1/(deltaS*deltaS);
	REAL alpha2 = 0.5*deltaT*sigma*sigma *S2*S2/(deltaS*deltaS);
	REAL beta1 = 0.5*deltaT*r*S1/deltaS;
	REAL beta2 = 0.5*deltaT*r*S2/deltaS;
	d_a1[t_id+b_id*blockDim.x] = -alpha1 + beta1;
	d_b1[t_id+b_id*blockDim.x] = 1.0f + 2*alpha1 +r*deltaT;
	d_c1[t_id+b_id*blockDim.x] = -alpha1 - beta1;
	//initialise the payoff as the right hand of tridiagonal system d_d
	//vanilla European call option	
	d1[t_id] = max(S1-K,0.0);
        d2[t_id] = max(S2-K,0.0);

    
    if(t_id==0) d_a1[b_id*blockDim.x]=0.0;
    __syncthreads();

	
          
    // main time-marching
    for(int k=0;k<ITERATION;k++){

     a2[t_id] =  -alpha2 + beta2;
     b2[t_id] =  1.0f + 2*alpha2 +r*deltaT;
     c2[t_id] =  -alpha2 - beta2;
     if(t_id==blockDim.x-1)    c2[t_id]=0.0; 


     // explicit and last row transformation 
     int up=0,down=0;
     if(t_id>0) up=t_id-1;
     if(t_id<blockDim.x-1) down=t_id+1;
     __syncthreads();

     REAL d1_temp=-d_a1[t_id+b_id*blockDim.x]*d2[up]+(2-d_b1[t_id+b_id*blockDim.x])*d1[t_id]-d_c1[t_id+b_id*blockDim.x]*d2[t_id];
     REAL d2_temp=-a2[t_id]*d1[t_id]+(2-b2[t_id])*d2[t_id]-c2[t_id]*d1[down];
     __syncthreads();
     d1[t_id]=d1_temp;d2[t_id]=d2_temp;
     if(t_id==blockDim.x-1) {
	 REAL cmax= -alpha2 - beta2;   
     	d2[t_id]+=-cmax*(d_upperbound[k+1]+d_upperbound[k]);   
	}

    __syncthreads();


    
// step 1: forward elimination by CR, only once

        // reduced to half size of the system,
	//d_a1[i]*x[2i-1]+d_b1[i]*x[2i]+d_c1[i]*x[2i+1]=d1[i]
        //a2[i]*x[2i]+b2[i]*x[2i+1]+c2[i]*x[2i+2]=d2[i]   <<<<only update the even equations 
        //d_a1[i+1]*x[2i+1]+d_b1[i+1]*x[2i+2]+d_c1[i+1]*x[2i+3]=d1[i+1]

	
	REAL r_up = a2[t_id] *__rcp( d_b1[t_id+b_id*blockDim.x]);
	REAL r_down =0.0;
	if (t_id == blockDim.x-1){
		b2[t_id] = b2[t_id] - d_c1[t_id+b_id*blockDim.x] * r_up  ;
		d2[t_id] = d2[t_id] - d1[t_id] * r_up  ;
		a2[t_id] = -d_a1[t_id+b_id*blockDim.x] * r_up;
	}
	else{	
 		r_down= c2[t_id]  *__rcp(d_b1[t_id+1+b_id*blockDim.x]);
		b2[t_id] = b2[t_id] - d_c1[t_id+b_id*blockDim.x] * r_up  - d_a1[t_id+1+b_id*blockDim.x] * r_down ;
		d2[t_id] = d2[t_id] - d1[t_id] * r_up  - d1[t_id+1] * r_down ;
		a2[t_id] = -d_a1[t_id+b_id*blockDim.x] * r_up;
		c2[t_id] = -d_c1[t_id+1+b_id*blockDim.x] * r_down ;
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
     	d1[t_id] = (d1[t_id] - d_c1[t_id+b_id*blockDim.x]*d2[t_id])*__rcp(d_b1[t_id+b_id*blockDim.x]);
    else
        d1[t_id] = (d1[t_id] - d_a1[t_id+b_id*blockDim.x]*d2[t_id-1] - d_c1[t_id+b_id*blockDim.x]*d2[t_id])*__rcp(d_b1[t_id+b_id*blockDim.x]);
    __syncthreads();
    
    }

    d_x[2*t_id+b_id*2*blockDim.x] = d1[t_id];
    d_x[2*t_id+1+b_id*2*blockDim.x] = d2[t_id];
}








// get the price at time 0
template <typename REAL>
REAL getPrice(REAL s,REAL ds,REAL *V)
{
   int i = (int)(s/ds);
   REAL w1 = (s-ds*i)/ds, w0 = 1.0-w1;
   return w1*V[i] + w0*V[i-1];
}











int main()
{
	int num_options,num_dissteps;
	int num_blocks,num_threads;

// record time
  	float *milli;
	milli = (float *)malloc(sizeof(float)*2);
  	cudaEvent_t start, stop;
  	cudaEventCreate(&start);
  	cudaEventCreate(&stop);

// store the results
	std::fstream out;
    out.open("Pricing by BS model.txt",std::ios_base::out);

//Pricing 1024 European call options 
    num_options=1024;
    
	for(int process=0;process<2;process++){
	if(process==0)
	{

	
// 1)
// Discretise the underlying asset price S into 514 points, time steps=200, predefined
// by implicit method, single precision, using PCR+Thomas algorithm to compute the tridiagonal systems

		num_dissteps=512;  //exclude end points Smin=0.0 and Smax=200.0

        // Set the parameters and compute the corresponding upper boundary. (we will treat Smin=0.0 in this case)
		float r=0.05, sigma=0.2, T=1.0/12, K=100.0, Smax=200.0;
		float *upperbound,*d_upperbound;
		upperbound= (float *)malloc((TIMESTEPS+1)* sizeof(float));
		checkCudaErrors( cudaMalloc( (void**) &d_upperbound,(TIMESTEPS+1)*sizeof(float)));
		for(int i=0;i<=TIMESTEPS;i++)
			upperbound[i]=(Smax-K)*exp(-r*T/TIMESTEPS*i);  //vanilla European call option	
   	 	checkCudaErrors(cudaMemcpy(d_upperbound, upperbound, (TIMESTEPS+1)* sizeof(float), cudaMemcpyHostToDevice));

   	 	// allocate memory space for global array of solutions
		int memsize=num_options*num_dissteps*sizeof(float);
   		float* d_x;
		checkCudaErrors( cudaMalloc( (void**) &d_x,memsize));

		cudaEventRecord(start);
		num_threads=256;num_blocks=num_options/8;

         // perform the main time-marching of finite difference method, implicit scheme

    	implicit_pcrThomas<<<num_blocks,num_threads>>>(sigma, Smax,K,T, r, d_x,d_upperbound);  
        // copy results from device to host
		float* x;
		x = (float *)malloc(memsize);       
   		checkCudaErrors(cudaMemcpy(x, d_x, memsize, cudaMemcpyDeviceToHost));

		cudaEventRecord(stop);
  		cudaEventSynchronize(stop);
  		cudaEventElapsedTime(&milli[0], start, stop);

        if(out.is_open()){
			out<<"European call option: r=0.05, sigma=0.2, T=1.0/12, K=100.0, Smax=200.0"<<std::endl;
			out<<"finite difference methos:implicit scheme, timesteps=200, discreticise size=514"<<std::endl;
			out<<"single precision, use 'pcr_Thomas' algorithm by shuffle' to solve the tridiagonal systems"<<std::endl;
       		for(int i=0;i<num_dissteps;i++)
				out<<"x["<<i<<"]="<<x[i]<<std::endl;
		}

        // Get the price by interpolation method
		float S0=103.0,ds=Smax/(num_dissteps+1);
		std::cout<<"implicit scheme, single precision, discretising in 514 steps"<<std::endl;
		std::cout<<"The price we get at time 0 when spot price is 103.0:"<<getPrice(S0,ds,x)<<std::endl;
		std::cout<<"executing time for one option:"<<milli[0]/num_options<<std::endl;
		// cleanup memory
    	checkCudaErrors(cudaFree(d_x));
    	checkCudaErrors(cudaFree(d_upperbound));
    	free(x);    	free(upperbound);
	}


	else
	{

		// 2)
		// Discretise the underlying asset price S into 2048 points (excluding Smin and Smax), time steps=200, predefined
		// by CN method, single precision, using PCR+CR algorithm to compute the tridiagonal systems
		
	num_dissteps=2048; //excluding end points Smin=0.0, Smax=200.0

       // Set the parameters and compute the corresponding upper boundary. (we will treat Smin=0.0 in this case)
		double r=0.05, sigma=0.2, T=1.0/12, K=100.0, Smax=200.0;
		double *upperbound,*d_upperbound;
		upperbound= (double *)malloc((TIMESTEPS+1)* sizeof(double));
		checkCudaErrors( cudaMalloc( (void**) &d_upperbound,(TIMESTEPS+1)*sizeof(double)));
		for(int i=0;i<=TIMESTEPS;i++)
			upperbound[i]=(Smax-K)*exp(-r*T/TIMESTEPS*i);  //vanilla European call option	
    		checkCudaErrors(cudaMemcpy(d_upperbound, upperbound, (TIMESTEPS+1)* sizeof(double), cudaMemcpyHostToDevice));


        // allocate memory space for global arrays of coefficients
		int memsize=num_options*num_dissteps*sizeof(double);
   		double* d_a1; double* d_b1; double* d_c1; double* d_x;
		checkCudaErrors( cudaMalloc( (void**) &d_a1,memsize/2));
		checkCudaErrors( cudaMalloc( (void**) &d_b1,memsize/2));
		checkCudaErrors( cudaMalloc( (void**) &d_c1,memsize/2));
		checkCudaErrors( cudaMalloc( (void**) &d_x,memsize));

		num_blocks=num_options; num_threads=num_dissteps/2;
		cudaEventRecord(start);	
        // perform the CN implicit scheme of finite difference method
    		CN_crpcr<<<num_blocks,num_threads,(num_dissteps+1)*5*sizeof(float)>>>(sigma, Smax,K,T, r,d_a1, d_b1, d_c1, d_x,d_upperbound);

        // copy results from device to host
		double* x;
		x = (double *)malloc(memsize);     
    	checkCudaErrors(cudaMemcpy(x, d_x, memsize, cudaMemcpyDeviceToHost));

		cudaEventRecord(stop);
  		cudaEventSynchronize(stop);
  		cudaEventElapsedTime(&milli[1], start, stop);

        if(out.is_open()){
			out<<"European call option: r=0.05, sigma=0.2, T=1.0/12, K=100.0, Smax=200.0"<<std::endl;
			out<<"finite difference methos:CN scheme, timesteps=200, discretising size=2050"<<std::endl;
			out<<"double precision, use b 'cr+pcr algorithm' to solve the tridiagonal systems"<<std::endl;
        	for(int i=0;i<num_dissteps;i++)
				out<<"x["<<i<<"]="<<x[i]<<std::endl;
		}
		
// Get the price by interpolation method                        
		double S0=103.0,ds=Smax/(num_dissteps+1);
		std::cout<<"CN scheme, double precision, discretising in 2050 steps"<<std::endl;
		std::cout<<"The price we get at time 0 when spot price is 103.0:"<<getPrice(S0,ds,x)<<std::endl;
		std::cout<<"executing time for one option (ms):"<<milli[1]/num_options<<std::endl;

    	// cleanup memory
		checkCudaErrors(cudaFree(d_a1));
    		checkCudaErrors(cudaFree(d_b1));
    		checkCudaErrors(cudaFree(d_c1));
    		checkCudaErrors(cudaFree(d_x));
    		checkCudaErrors(cudaFree(d_upperbound));
   		free(x);free(upperbound);

	}

    }
	out.close();
	free(milli);

return 0;
    
}
