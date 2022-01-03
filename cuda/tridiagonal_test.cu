
#include "tridiagonal_solver.h"
#include <chrono>
#include <fstream>
using namespace std;

template <typename REAL>
void generateTrisystem(REAL r,REAL sigma,REAL T,REAL K,REAL Smax,REAL *a,REAL *b,REAL *c,REAL *d,REAL*x,int n)
{


	REAL deltaS = Smax/ (n+1);   
	REAL deltaT = T/  TIMESTEPS;
//S from 1* deltaS to blockDim.x* deltaS (excluding Smin=0, Smax=(blockDim  .x+1)* deltaS )
	for(int i=0;i<n;i++)
	{
	REAL S = (i+1)* deltaS;

	REAL alpha = 0.5*deltaT*sigma*sigma *S*S/(deltaS*deltaS);
	REAL beta = 0.5*deltaT*r*S/deltaS;
	REAL aa=-alpha + beta;
	REAL bb = 1.0f + 2*alpha +r*deltaT;
	REAL cc = -alpha - beta;
	REAL dd = max(S-K,0.0);

	for(int j=0;j<SYSTEMS;j++)
	{
	a[i+j*n]=aa;
	b[i+j*n]=bb;
	c[i+j*n]=cc;
	d[i+j*n]=dd;
	x[i+j*n]=dd;
	}	
/*cout<<"host  original---"<<a[i]<<" "<<b[i]<<" "<<c[i]<<" "<<d[i]<<endl;*/
	}

}






int main()
{
	//int systemsize=512;
	//int systemsize=1024;
	int systemsize=2048;


//record the executing time of each algorithm
  	float *milli;
	milli = (float *)malloc(sizeof(float)*10);
  	cudaEvent_t start, stop;
  	cudaEventCreate(&start);
  	cudaEventCreate(&stop);

  	chrono::time_point<std::chrono::system_clock> cpustart, cpuend;
	chrono::duration<double> elapsed_seconds;

// calculate the maximum blocks and maximum systems
	milli = (float *)malloc(sizeof(float)*10);
	printf("execution time iterating %d steps (ms), system size=%d\n\n",ITERATION,systemsize);
	printf("   Thomas     CR        CR(NBC)       PCR       PCR+CR      PCR+Thomas\n");

//write the results in the txt file
	fstream out;



	for (int precision=0; precision<2; precision++) {



/*********************************************** single precision*********************************************************************/
/*************************************************************************************************************************************/
	if(precision==0){

//tridiagonal matrix(a,b,c), the right hand d, the left hand to be solved x	
	float* a; float* b; float* c; float* d;float *x;
	a = (float *)malloc(systemsize*SYSTEMS*sizeof(float));
	b = (float *)malloc(systemsize*SYSTEMS*sizeof(float));
	c = (float *)malloc(systemsize*SYSTEMS*sizeof(float));
	d = (float *)malloc(systemsize*SYSTEMS*sizeof(float));
	x = (float *)malloc(systemsize*SYSTEMS*sizeof(float));

//generate the tridiagonal system randomly
	float r=0.05,sigma=0.2,T=1.0/12,K=100.0,Smax=200.0;
	generateTrisystem(r,sigma,T,K,Smax,a,b,c,d,x,systemsize);

// open the file to be written in
 	if(systemsize==512)  out.open("system512SP.txt",ios_base::out);
		else if(systemsize==1024)  out.open("system1024SP.txt",ios_base::out);
		     else  out.open("system2048SP.txt",ios_base::out);

//////////////////////////////////////////////////////////////////
//		Thomas algorithm (serial version) 		//
//////////////////////////////////////////////////////////////////


  cpustart = chrono::system_clock::now();

	for(int i=0;i<ITERATION;i++)
		Thomas(a,b,c,x,systemsize);
  cpuend = chrono::system_clock::now();
  elapsed_seconds = (cpuend-cpustart)*1024;

   if(out.is_open()){
	out<<"**************Thomas************** "<<systemsize<<" x "<<systemsize<<endl;
        for(int i=0;i<systemsize;i++)
		out<<"x["<<i<<"]="<<x[i]<<endl;
        }


//allocate memory for device global arrays	
   	float* d_a; float* d_b; float* d_c; float* d_d;
	checkCudaErrors( cudaMalloc( (void**) &d_a,systemsize*SYSTEMS*sizeof(float)));
	checkCudaErrors( cudaMalloc( (void**) &d_b,systemsize*SYSTEMS*sizeof(float)));
	checkCudaErrors( cudaMalloc( (void**) &d_c,systemsize*SYSTEMS*sizeof(float)));
	checkCudaErrors( cudaMalloc( (void**) &d_d,systemsize*SYSTEMS*sizeof(float)));

//copy from host to device
 	checkCudaErrors(cudaMemcpy(d_a, a, systemsize *SYSTEMS* sizeof(float),cudaMemcpyHostToDevice));
 	checkCudaErrors(cudaMemcpy(d_b, b, systemsize *SYSTEMS* sizeof(float),cudaMemcpyHostToDevice));
 	checkCudaErrors(cudaMemcpy(d_c, c, systemsize *SYSTEMS* sizeof(float),cudaMemcpyHostToDevice));



//set the bank size of shared memory as four bytes
       cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);



//////////////////////////////////////////////////////////////////
//		Cyclic reduction 			        //
//////////////////////////////////////////////////////////////////

// blocksize=systemsize/2, one block one system
// 1)by shared memory with bank conflicts
 	checkCudaErrors(cudaMemcpy(d_d, d, systemsize * SYSTEMS*sizeof(float),cudaMemcpyHostToDevice));
	cudaEventRecord(start);
	for(int i=0;i<ITERATION;i++)
		crKernel<<<SYSTEMS,systemsize/2,systemsize*5*sizeof(float)>>>(d_a, d_b, d_c, d_d);
	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&milli[0], start, stop);
 	checkCudaErrors( cudaMemcpy(x, d_d,systemsize* SYSTEMS * sizeof(float), cudaMemcpyDeviceToHost) );

// 2)by shared memory with no bank conflicts
 	checkCudaErrors(cudaMemcpy(d_d, d, systemsize *SYSTEMS* sizeof(float),cudaMemcpyHostToDevice));
	cudaEventRecord(start);
	for(int i=0;i<ITERATION;i++)
		crNBCKernel<<<SYSTEMS,systemsize/2,systemsize*5*sizeof(float)*33/32>>>(d_a, d_b, d_c, d_d);
	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&milli[1], start, stop);
 	checkCudaErrors( cudaMemcpy(x, d_d,systemsize*SYSTEMS * sizeof(float), cudaMemcpyDeviceToHost) );
        if(out.is_open()){
	out<<"**************CR**************"<<systemsize<<" x "<<systemsize<<endl;
        for(int i=0;i<systemsize;i++)
		out<<"x["<<i<<"]="<<x[i]<<endl;
	}


//////////////////////////////////////////////////////////////////
//		Parallel cyclic reduction		        //
//////////////////////////////////////////////////////////////////


//by shared memory 
// 1)blocksize = systemsize, one block one system
	if (systemsize<=1024){
	checkCudaErrors(cudaMemcpy(d_d, d, systemsize *SYSTEMS* sizeof(float),cudaMemcpyHostToDevice));
	cudaEventRecord(start);
	for(int i=0;i<ITERATION;i++)
		pcrKernel<<<SYSTEMS,systemsize,(systemsize+1)*4*sizeof(float)>>>(d_a, d_b, d_c, d_d);
	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&milli[2], start, stop);
 	checkCudaErrors( cudaMemcpy(x, d_d,systemsize *SYSTEMS* sizeof(float), cudaMemcpyDeviceToHost) );
	
	}

// 2)divide the system into two blocks, blocksize=systemsize/2, two blocks one system
	if (systemsize==2048){
	checkCudaErrors(cudaMemcpy(d_d, d, systemsize *SYSTEMS* sizeof(float),cudaMemcpyHostToDevice));
	float *d_z,*d_alpha;
	checkCudaErrors( cudaMalloc( (void**) &d_z,SYSTEMS*2*sizeof(float)));
	checkCudaErrors( cudaMalloc( (void**) &d_alpha,SYSTEMS*2*sizeof(float)));
	cudaEventRecord(start);   
	for(int i=0;i<ITERATION;i++)
		pcrX2Kernel<<<SYSTEMS*2,systemsize/2,(systemsize/2+1)*5*sizeof(float)>>>(d_a, d_b, d_c, d_d, d_z,d_alpha);
	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&milli[2], start, stop);
 	checkCudaErrors( cudaMemcpy(x, d_d,systemsize *SYSTEMS* sizeof(float), cudaMemcpyDeviceToHost) );
        checkCudaErrors( cudaFree(d_z) );
	}
        

        if(out.is_open()){
	out<<"**************PCR**************"<<systemsize<<" x "<<systemsize<<endl;
        for(int i=0;i<systemsize;i++)
		out<<"x["<<i<<"]="<<x[i]<<endl;
	}



/////////////////////////////////////////////////////////////////////////
//	Parallel cyclic reduction + cyclic reduction		       //
/////////////////////////////////////////////////////////////////////////

// blocksize=systemsize/2, one block one system
 	checkCudaErrors(cudaMemcpy(d_d, d, systemsize *SYSTEMS* sizeof(float),cudaMemcpyHostToDevice));
	cudaEventRecord(start);
	pcr_cr_Kernel<<<SYSTEMS,systemsize/2,(systemsize+2)*4*sizeof(float)>>>(d_a, d_b, d_c, d_d);
	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&milli[3], start, stop);
 	checkCudaErrors( cudaMemcpy(x, d_d,systemsize * SYSTEMS*sizeof(float), cudaMemcpyDeviceToHost) );
        if(out.is_open()){
	out<<"**************PCR+CR**************"<<systemsize<<" x "<<systemsize<<endl;
        for(int i=0;i<systemsize;i++)
		out<<"x["<<i<<"]="<<x[i]<<endl;
	}
   


/////////////////////////////////////////////////////////////////////////
//		Parallel cyclic reduction + Thomas		       //
/////////////////////////////////////////////////////////////////////////

// one warp one system
if(systemsize==512)
    {
	checkCudaErrors(cudaMemcpy(d_d, d, systemsize *SYSTEMS*sizeof(float),cudaMemcpyHostToDevice));
	cudaEventRecord(start);
			pcr_Thomas_warp1<<<SYSTEMS/8,256>>>(d_a, d_b, d_c, d_d); 
	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&milli[4], start, stop);
	
 	checkCudaErrors( cudaMemcpy(x, d_d,systemsize *SYSTEMS* sizeof(float), cudaMemcpyDeviceToHost) );
     }
 if(systemsize==1024)
     {
	checkCudaErrors(cudaMemcpy(d_d, d, systemsize *SYSTEMS*sizeof(float),cudaMemcpyHostToDevice));
	cudaEventRecord(start);
			pcr_Thomas_warp2<<<SYSTEMS/8,256>>>(d_a, d_b, d_c, d_d); 
	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&milli[4], start, stop);
	
 	checkCudaErrors( cudaMemcpy(x, d_d,systemsize *SYSTEMS* sizeof(float), cudaMemcpyDeviceToHost) );
     }

    if(systemsize==2048)
   {
	checkCudaErrors(cudaMemcpy(d_d, d, systemsize *SYSTEMS*sizeof(float),cudaMemcpyHostToDevice));
	cudaEventRecord(start);
			pcr_Thomas_warp3<<<SYSTEMS/8,256>>>(d_a, d_b, d_c, d_d); 
	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&milli[4], start, stop);
	
 	checkCudaErrors( cudaMemcpy(x, d_d,systemsize *SYSTEMS* sizeof(float), cudaMemcpyDeviceToHost) );
   }


        if(out.is_open()){
	out<<"**************PCR+Thomas**************"<<systemsize<<" x "<<systemsize<<endl;
        for(int i=512*systemsize;i<513*systemsize;i++)
		out<<"x["<<i<<"]="<<x[i]<<endl;
	}

        
//free the memory
	free(a);free(b);free(c);free(d);free(x);

	checkCudaErrors( cudaFree(d_a) );
	checkCudaErrors( cudaFree(d_b) );
	checkCudaErrors( cudaFree(d_c) );
	checkCudaErrors( cudaFree(d_d) );


	out.close();
//showing executing time
	printf("SP:%f  %f   %f     %f    %f         %f\n", elapsed_seconds,milli[0],milli[1],milli[2],milli[3],milli[4]);	
	//printf("num of systems:1 %d         %d          %d           %d             %d\n", maxsystems[0],maxsystems[1],maxsystems[2],maxsystems[3],maxsystems[4]);	
	printf("speed up: %f   %f     %f    %f         %f\n",elapsed_seconds/milli[0]*SYSTEMS,elapsed_seconds/milli[1]*SYSTEMS,elapsed_seconds/milli[2]*SYSTEMS,elapsed_seconds/milli[3]*SYSTEMS,elapsed_seconds/milli[4]*SYSTEMS);						
 	}

/***********************************************************double precision**********************************************************/
/*************************************************************************************************************************************/
	else{

//tridiagonal matrix(a,b,c), the right hand d, the left hand to be solved x	
	double* a; double* b; double* c; double* d;double *x;
	a = (double *)malloc(systemsize*SYSTEMS*sizeof(double));
	b = (double *)malloc(systemsize*SYSTEMS*sizeof(double));
	c = (double *)malloc(systemsize*SYSTEMS*sizeof(double));
	d = (double *)malloc(systemsize*SYSTEMS*sizeof(double));
	x = (double *)malloc(systemsize*SYSTEMS*sizeof(double));

//generate the tridiagonal system randomly
	double r=0.05,sigma=0.2,T=1.0/12,K=100.0,Smax=200.0;
	generateTrisystem(r,sigma,T,K,Smax,a,b,c,d,x,systemsize);


// open the file to be written in
 	if(systemsize==512)  out.open("system512DP.txt",ios_base::out);
		else if(systemsize==1024)  out.open("system1024DP.txt",ios_base::out);
		     else  out.open("system2048DP.txt",ios_base::out);

//////////////////////////////////////////////////////////////////
//		Thomas algorithm (serial version) 		//
//////////////////////////////////////////////////////////////////



  cpustart = chrono::system_clock::now();

	for(int i=0;i<ITERATION;i++)
		Thomas(a,b,c,x,systemsize);
  cpuend = chrono::system_clock::now();
  elapsed_seconds = (cpuend-cpustart)*1024;

        if(out.is_open()){
	out<<"**************Thomas**************"<<systemsize<<" x "<<systemsize<<endl;
        for(int i=0;i<systemsize;i++)
		out<<"x["<<i<<"]="<<x[i]<<endl;
	}



//allocate memory for device global arrays	
   	double* d_a; double* d_b; double* d_c; double* d_d;
	checkCudaErrors( cudaMalloc( (void**) &d_a,systemsize*SYSTEMS*sizeof(double)));
	checkCudaErrors( cudaMalloc( (void**) &d_b,systemsize*SYSTEMS*sizeof(double)));
	checkCudaErrors( cudaMalloc( (void**) &d_c,systemsize*SYSTEMS*sizeof(double)));
	checkCudaErrors( cudaMalloc( (void**) &d_d,systemsize*SYSTEMS*sizeof(double)));

//copy from host to device
 	checkCudaErrors(cudaMemcpy(d_a, a, systemsize*SYSTEMS * sizeof(double),cudaMemcpyHostToDevice));
 	checkCudaErrors(cudaMemcpy(d_b, b, systemsize*SYSTEMS * sizeof(double),cudaMemcpyHostToDevice));
 	checkCudaErrors(cudaMemcpy(d_c, c, systemsize*SYSTEMS * sizeof(double),cudaMemcpyHostToDevice));


//set the bank size of shared memory as eight byte
      cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);


//////////////////////////////////////////////////////////////////
//		Cyclic reduction 			        //
//////////////////////////////////////////////////////////////////

// blocksize=systemsize, one block one system
// 1)by shared memory with bank conflicts
if(systemsize<=1024){
 	checkCudaErrors(cudaMemcpy(d_d, d, systemsize*SYSTEMS* sizeof(double),cudaMemcpyHostToDevice));
	cudaEventRecord(start);
	for(int i=0;i<ITERATION;i++)
		crKernel<<<SYSTEMS,systemsize/2,systemsize*5*sizeof(double)>>>(d_a, d_b, d_c, d_d);
	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&milli[0], start, stop);
 	checkCudaErrors( cudaMemcpy(x, d_d,systemsize *SYSTEMS* sizeof(double), cudaMemcpyDeviceToHost) );


// 2)by shared memory with no bank conflicts
 	checkCudaErrors(cudaMemcpy(d_d, d, systemsize *SYSTEMS* sizeof(double),cudaMemcpyHostToDevice));
	cudaEventRecord(start);
	for(int i=0;i<ITERATION;i++)
		crNBCKernel<<<SYSTEMS,systemsize/2,systemsize*5*sizeof(double)*33/32>>>(d_a, d_b, d_c, d_d);
	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&milli[1], start, stop);
 	checkCudaErrors( cudaMemcpy(x, d_d,systemsize *SYSTEMS* sizeof(double), cudaMemcpyDeviceToHost) );

}
if(systemsize==2048){

// 3)split into two blocks, blocksize=systemsize/2, two blocks one system
 	checkCudaErrors(cudaMemcpy(d_d, d, systemsize *SYSTEMS* sizeof(double),cudaMemcpyHostToDevice));
	double *d_y,*d_z,*d_alpha;
	checkCudaErrors( cudaMalloc( (void**) &d_y,systemsize*SYSTEMS*sizeof(double)));
	checkCudaErrors( cudaMalloc( (void**) &d_z,systemsize*SYSTEMS*sizeof(double)));
	checkCudaErrors( cudaMalloc( (void**) &d_alpha,SYSTEMS*sizeof(double)));
	cudaEventRecord(start);
	for(int i=0;i<ITERATION;i++)
		crNBCX2Kernel<<<SYSTEMS*2,systemsize/2,systemsize/2*5*sizeof(double)*33/32>>>(d_a, d_b, d_c,d_d,d_y,d_z,d_alpha);
	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&milli[1], start, stop);
	milli[0]=milli[1];
 	checkCudaErrors( cudaMemcpy(x, d_d,systemsize *SYSTEMS* sizeof(double), cudaMemcpyDeviceToHost) );
	
	checkCudaErrors( cudaFree(d_y) );
	checkCudaErrors( cudaFree(d_z) );
}
        if(out.is_open()){
	out<<"**************CR**************"<<systemsize<<" x "<<systemsize<<endl;
        for(int i=0;i<systemsize;i++)
		out<<"x["<<i<<"]="<<x[i]<<endl;
	}

//////////////////////////////////////////////////////////////////
//		Parallel cyclic reduction			//
//////////////////////////////////////////////////////////////////
 	
 //by shared memory
 // 1) blocksize=systemsize, one block one system
	if (systemsize<=1024){
	checkCudaErrors(cudaMemcpy(d_d, d, systemsize * SYSTEMS*sizeof(double),cudaMemcpyHostToDevice));
	cudaEventRecord(start);
	for(int i=0;i<ITERATION;i++)
		pcrKernel<<<SYSTEMS,systemsize,(systemsize+1)*4*sizeof(double)>>>(d_a, d_b, d_c, d_d);
	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&milli[2], start, stop);
 	checkCudaErrors( cudaMemcpy(x, d_d,systemsize*SYSTEMS * sizeof(double), cudaMemcpyDeviceToHost) );
	}

// 2) divide the system into two blocks, blocksize=systemsize/2,  two blocks per system
	if (systemsize==2048){
	double *d_z,*d_alpha;
	checkCudaErrors( cudaMalloc( (void**) &d_z,SYSTEMS*2*sizeof(double)));
	checkCudaErrors( cudaMalloc( (void**) &d_alpha,SYSTEMS*2*sizeof(double)));
	checkCudaErrors(cudaMemcpy(d_d, d, systemsize * SYSTEMS*sizeof(double),cudaMemcpyHostToDevice));
	cudaEventRecord(start);
	for(int i=0;i<ITERATION;i++)
		pcrX2Kernel<<<SYSTEMS*2,systemsize/2,(systemsize/2+1)*5*sizeof(double)>>>(d_a, d_b, d_c, d_d, d_z,d_alpha);
	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&milli[2], start, stop);
 	checkCudaErrors( cudaMemcpy(x, d_d,systemsize*SYSTEMS * sizeof(double), cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaFree(d_z) );
	}

        if(out.is_open()){
	out<<"**************PCR**************"<<systemsize<<" x "<<systemsize<<endl;
        for(int i=0;i<systemsize;i++)
		out<<"x["<<i<<"]="<<x[i]<<endl;
	}


/////////////////////////////////////////////////////////////////////////
//	Parallel cyclic reduction + cyclic reduction		       //
/////////////////////////////////////////////////////////////////////////

// blocksize=systemsize/2, one block one system
 	checkCudaErrors(cudaMemcpy(d_d, d, systemsize * SYSTEMS*sizeof(double),cudaMemcpyHostToDevice));
	cudaEventRecord(start);
	if(systemsize<=1024)
	pcr_cr_Kernel<<<SYSTEMS,systemsize/2,(systemsize+2)*4*sizeof(double)>>>(d_a, d_b, d_c, d_d);
	else
	pcr_cr_LKernel<<<SYSTEMS,systemsize/2,(systemsize/2+1)*5*sizeof(double)>>>(d_a, d_b, d_c, d_d);
	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&milli[3], start, stop);
 	checkCudaErrors( cudaMemcpy(x, d_d,systemsize * SYSTEMS*sizeof(double), cudaMemcpyDeviceToHost) );
        if(out.is_open()){
	out<<"**************PCR+CR**************"<<systemsize<<" x "<<systemsize<<endl;
        for(int i=0;i<systemsize;i++)
		out<<"x["<<i<<"]="<<x[i]<<endl;
	}


/////////////////////////////////////////////////////////////////////////
//		Parallel cyclic reduction + Thomas		       //
/////////////////////////////////////////////////////////////////////////

// calculate one system within the warp 


if(systemsize==512) 
{
   
	checkCudaErrors(cudaMemcpy(d_d, d, systemsize *SYSTEMS*sizeof(double),cudaMemcpyHostToDevice));
	cudaEventRecord(start);
			pcr_Thomas_warp1<<<SYSTEMS/8,256>>>(d_a, d_b, d_c, d_d); 
	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&milli[4], start, stop);
	
 	checkCudaErrors( cudaMemcpy(x, d_d,systemsize *SYSTEMS* sizeof(double), cudaMemcpyDeviceToHost) );
     }
if(systemsize==1024)
     {
	checkCudaErrors(cudaMemcpy(d_d, d, systemsize *SYSTEMS*sizeof(double),cudaMemcpyHostToDevice));
	cudaEventRecord(start);
			pcr_Thomas_warp2<<<SYSTEMS/8,256>>>(d_a, d_b, d_c, d_d); 
	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&milli[4], start, stop);
	
 	checkCudaErrors( cudaMemcpy(x, d_d,systemsize *SYSTEMS* sizeof(double), cudaMemcpyDeviceToHost) );
     }

    if(systemsize==2048)
     {
	checkCudaErrors(cudaMemcpy(d_d, d, systemsize *SYSTEMS*sizeof(double),cudaMemcpyHostToDevice));
	cudaEventRecord(start);
			pcr_Thomas_warp3<<<SYSTEMS/8,256>>>(d_a, d_b, d_c, d_d); 
	cudaEventRecord(stop);
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&milli[4], start, stop);
	
 	checkCudaErrors( cudaMemcpy(x, d_d,systemsize *SYSTEMS* sizeof(double), cudaMemcpyDeviceToHost) );
     }


        if(out.is_open()){
	out<<"**************PCR+Thomas**************"<<systemsize<<" x "<<systemsize<<endl;
        for(int i=0;i<systemsize;i++)
		out<<"x["<<i<<"]="<<x[i]<<endl;
	}
	out.close();



	free(a);free(b);free(c);free(d);free(x);

	checkCudaErrors( cudaFree(d_a) );
	checkCudaErrors( cudaFree(d_b) );
	checkCudaErrors( cudaFree(d_c) );
	checkCudaErrors( cudaFree(d_d) );

//showing executing time 
	printf("DP:%f  %f   %f     %f    %f         %f\n", elapsed_seconds,milli[0],milli[1],milli[2],milli[3],milli[4]);		
	printf("speed up: %f   %f     %f    %f         %f\n",elapsed_seconds/milli[0]*SYSTEMS,elapsed_seconds/milli[1]*SYSTEMS,elapsed_seconds/milli[2]*SYSTEMS,elapsed_seconds/milli[3]*SYSTEMS,elapsed_seconds/milli[4]*SYSTEMS);						
	}

}

	
	free(milli);	

// CUDA exit -- needed to flush printf write buffer	
	cudaDeviceReset();
	return 0;
}
