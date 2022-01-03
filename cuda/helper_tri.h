//////////////////////////////////////////////////////////////////////////
//                        64 bit shuffle command                        //
//////////////////////////////////////////////////////////////////////////
/*
__forceinline__ __device__ double __shfl_up(double x,  int s) {
  int lo, hi;
  //http://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#axzz4nOFS1LMu
  //split the double number into 2 32bit registers
  asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x) );
  //shuffle the two 32bit registers
  lo = __shfl_up(lo,s,32); //????????????????why I have to use (int,int,int)version,the third argument? width of warp??
  hi = __shfl_up(hi,s,32); //or else overload function,couldn't find the best match????????????????????????????????????
  //recreate the 64bit number
  asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo), "r"(hi) );

  return x;
}
  
__forceinline__ __device__ double __shfl_down(double x,  int s) {
  int lo, hi;
  asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x) );
  lo = __shfl_down(lo,s,32);
  hi = __shfl_down(hi,s,32);
  asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo), "r"(hi) );
  return x;
}

__forceinline__ __device__ double __shfl_xor(double x, int s) {
  int lo, hi;
  asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x) );
  lo = __shfl_xor(lo,s,32);
  hi = __shfl_xor(hi,s,32);
  asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo), "r"(hi) );
  return x;
}
*/

//////////////////////////////////////////////////////////////////////////
//                         define reciprocals                           //
//////////////////////////////////////////////////////////////////////////


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
