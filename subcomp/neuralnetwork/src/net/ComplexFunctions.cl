inline float2 cadd(float2 a, float2 b)
{
    return (float2)(a.x + b.x, a.y + b.y);
}

inline float2 caddr(float2 a, float r)
{
    return (float2)(a.x + r, a.y);
}

inline float2 cradd(float r, float2 a)
{
    return (float2)(a.x + r, a.y);
}

inline float2 csub(float2 a, float2 b)
{
    return (float2)(a.x - b.x, a.y - b.y);
}

inline float2 cmul(float2 a, float2 b)
{
    return (float2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

inline float real(float2 a)
{
    return a.x;
}

inline float imag(float2 a)
{
    return a.y;
}

inline float cabs(float2 a)
{
    return hypot(a.x, a.y); //square root of x^2 + y^2
}

// f(z)=tanh(z) = [sinh(2x) + j sin(2y)] /[cos(2y) + cosh(2y)]
// f(z)=a*tanh(b*z)
inline float2 ctanh(float2 z, float a, float b)
{  
  float re2 = b*2.f * z.x;
  float im2 = b*2.f * z.y; 
  const float limit = log((float)INT_MAX);

  if (fabs(re2) > limit)
  {
 //   printf("overflow tanh fabs(re2) = %f, limit = %f\n", fabs(re2), limit);
    return (float2)((re2 > 0 ? 1.f : -1.f), 0.f);
  }
  else
  {
    float den = cosh(re2) + cos(im2);    
    return (float2)(a*sinh(re2)/den, a*sin(im2)/den);
  }
}

// Ux = 2*(1+cosh(2x)*cos(2y))/(cosh(2x)+cos(2y))^2
float complextanhDerUx(float2 z, float a, float b)
{
    float re2 = b*2.f * z.x;
    float im2 = b*2.f * z.y;
    const float limit = log((float)INT_MAX);
    if (fabs(re2) > limit)
    {
        return (re2 > 0 ? 1.f : -1.f);
    }
    else
    {
        float den = cosh(re2) + cos(im2);
        return 2.*a*(1.+cosh(re2)*cos(im2))/(den*den);
    }
}

// Uy = 2*sinh(2x)*sin(2y))/(cosh(2x)+cos(2y))^2
float complextanhDerUy(float2 z, float a, float b)
{
    float re2 = b*2.f * z.x;
    float im2 = b*2.f * z.y;
    const float limit = log((float)INT_MAX);
    if (fabs(re2) > limit)
    {
        return (re2 > 0 ? 1.f : -1.f);
    }
    else
    {
        float den = cosh(re2) + cos(im2);
        return 2.*a*sinh(re2)*sin(im2)/(den*den);
    }
}

inline float2  cdiv(float2 nom, float2 den)
{
    const float n = den.x*den.x + den.y*den.y;
    return (float2)((nom.x*den.x + nom.y*den.y)/n, (nom.y*den.x-nom.x*den.y)/n);
}

inline float2 csinh(float2 z)
{
    return (float2)(sinh(z.x)*cos(z.y), cosh(z.x)*sin(z.y));
}

inline float2 ccosh(float2 z)
{
    return (float2)(cosh(z.x)*cos(z.y), sinh(z.x)*sin(z.y));
}

float2 sigmoidImpl(float2 input)
{
    float output = 1.0f/(1.0+exp(-input.x));
    return (float2)(output,0);
}


float2 sigmoidDerImpl(float2 input)
{
    return (float2)(input.x*(1-input.x),0);
}
// a*tanh(b*z) , b - gain coeff, a - saturation amplitude
__kernel void complextanh(global float* outRe, global float* outIm, const int outputOffset,
                          const global float* actPotRe, const global float* actPotIm, const int inputOffset,
                          const global float* biasRe, const global float* biasIm,
                          global float2* derU, global float2* derV,
                          unsigned int outChannel)
{
    const int gsiz = get_global_size(0);
    const int n = get_global_id(0);
    const int m = get_global_id(1);
    const int inIdx = outChannel*inputOffset + m*gsiz+n;
    const int outIdx = outChannel*outputOffset + m*gsiz+n;
    float saturation = 0.9f; //1.7159f;
    float gain = 0.1f; // 2/3


    float2 activationPotential = (float2)(actPotRe[inIdx]+biasRe[outChannel], actPotIm[inIdx]+biasIm[outChannel]);
    //printf("outIdx %d outChannel %d n,m (%d,%d) act=(%f,%f) activationPotential %f + j%f, bias = (%f, %f)\n", outIdx, outChannel, n, m, actPotRe[inIdx], actPotIm[inIdx], activationPotential.x, activationPotential.y, biasRe[outChannel], biasIm[outChannel]);
    float2  out = ctanh(activationPotential, saturation, gain);
    outRe[outIdx] = out.x;
    outIm[outIdx] = out.y;
  //  printf("outChannel %d outIdx %d output = %f + j%f\n", outChannel, outIdx, out.x, out.y);
    derU[outIdx].x = complextanhDerUx(activationPotential, saturation, gain);
    derU[outIdx].y = complextanhDerUy(activationPotential, saturation, gain);
    derV[outIdx].x = -derU[outIdx].y;
    derV[outIdx].y = derU[outIdx].x;
    //printf("outChannel %d outIdx %d derivation output = %f + j%f\n", outChannel, outIdx, out.x, out.y);
}

__kernel void sigmoid(global float* outRe, global float* outIm, const int outputOffset,
                      const global float* actPotRe, const global float* actPotIm, const int inputOffset,
                      const global float* biasRe, const global float* biasIm,
                      global float2* derU, global float2* derV,
                      unsigned int outChannel)
{
    const int gsiz = get_global_size(0);
    const int n = get_global_id(0);
    const int m = get_global_id(1);
    const int inIdx = outChannel*inputOffset + m*gsiz+n;
    const int outIdx = outChannel*outputOffset + m*gsiz+n;

    float2 activationPotential = (float2)(actPotRe[inIdx]+biasRe[outChannel], actPotIm[inIdx]+biasIm[outChannel]);
    float2  out = sigmoidImpl(activationPotential);
    outRe[outIdx] = out.x;
    outIm[outIdx] = out.y;
    out = sigmoidDerImpl(out);
    derU[outIdx].x = out.x;
    derU[outIdx].y = -out.y;
    derV[outIdx].x = out.y;
    derV[outIdx].y = out.x;
}

__kernel void realtanh(global float* outRe, global float* outIm, const int outputOffset,
                      const global float* actPotRe, const global float* actPotIm, const int inputOffset,
                      const global float* biasRe, const global float* biasIm,
                      global float2* derU, global float2* derV,
                      unsigned int outChannel)
{
    const int gsiz = get_global_size(0);
    const int n = get_global_id(0);
    const int m = get_global_id(1);
    const int inIdx = outChannel*inputOffset + m*gsiz+n;
    const int outIdx = outChannel*outputOffset + m*gsiz+n;

    outRe[outIdx] = tanh(actPotRe[inIdx]+biasRe[outChannel]);
    outIm[outIdx] = 0;
    derU[outIdx].x = 1 - outRe[outIdx]*outRe[outIdx];
    derU[outIdx].y = 0;
    derV[outIdx].x = 0;
    derV[outIdx].y = 0;
}


__kernel void fake(global float* outRe, global float* outIm, const int outputOffset,
                   const global float* actPotRe, const global float* actPotIm, const int inputOffset,
                   const global float* biasRe, const global float* biasIm,
                   global float2* derU, global float2* derV,
                   unsigned int outChannel)
{
    const int gsiz = get_global_size(0);
    const int n = get_global_id(0);
    const int m = get_global_id(1);
    const int inIdx = outChannel*inputOffset + m*gsiz+n;
    const int outIdx = outChannel*outputOffset + m*gsiz+n;

    float2 activationPotential = (float2)(actPotRe[inIdx]+biasRe[outChannel], actPotIm[inIdx]+biasIm[outChannel]);

    outRe[outIdx] = activationPotential.x;
    outIm[outIdx] = activationPotential.y;

    derU[outIdx].x = 1;
    derV[outIdx].x = 1;
}

//georgiou - f(z) = z/(c+(1/r)*|z|)
__kernel void georgiou(global float* outRe, global float* outIm, const int outputOffset,
                       const global float* actPotRe, const global float* actPotIm, const int inputOffset,
                       const global float* biasRe, const global float* biasIm,
                       global float2* derU, global float2* derV,
                       unsigned int outChannel)
{
    const int gsiz = get_global_size(0);
    const int n = get_global_id(0);
    const int m = get_global_id(1);
    const int inIdx = outChannel*inputOffset + m*gsiz+n;
    const int outIdx = outChannel*outputOffset + m*gsiz+n;
    const float c = 0.8;
    const float r = 1.5;

    const float2 activationPotential = (float2)(actPotRe[inIdx]+biasRe[outChannel], actPotIm[inIdx]+biasIm[outChannel]);
    const float absz = cabs(activationPotential);

    outRe[outIdx] = activationPotential.x/(c+(1.0f/r)*absz);
    outIm[outIdx] = activationPotential.y/(c+(1.0f/r)*absz);

    if(absz != 0)
    {
        derU[outIdx].x = r*(activationPotential.y*activationPotential.y + c*r*absz)/(absz*(c*r+absz)*(c*r+absz));
        derU[outIdx].y = -r*activationPotential.x*activationPotential.y/(absz*(c*r+absz)*(c*r+absz));
        derV[outIdx].x = -r*activationPotential.x*activationPotential.y/(absz*(c*r+absz)*(c*r+absz));
        derV[outIdx].y = r*(activationPotential.x*activationPotential.x + c*r*absz)/(absz*(c*r+absz)*(c*r+absz));
    }
    else
    {
        derU[outIdx].x = 1.0f/c;
        derU[outIdx].y = 0.0f;
        derV[outIdx].x = 0.0f;
        derV[outIdx].y = 1.0f/c;
    }
/*
    printf("outIdx = %d; outChannel = %d; (n,m) = (%d,%d); act=(%f,%f); aP = (%f,%f); out = (%f,%f)"
            " Ux = %f Uy = %f Vx = %f Vy = %f \n",
            outIdx,
            outChannel,
            n, m,
            actPotRe[inIdx], actPotIm[inIdx],
            activationPotential.x, activationPotential.y,
            outRe[outIdx], outIm[outIdx],
            derU[outIdx].x, derU[outIdx].y,
            derV[outIdx].x, derV[outIdx].y);
*/

}


// f(x+jy) = sinh(x+jy) = sinh(x)*cos(y)+jcosh(x)*sin(y)
__kernel void complexsinh(global float* outRe, global float* outIm, const int outputOffset,
                          const global float* actPotRe, const global float* actPotIm, const int inputOffset,
                          const global float* biasRe, const global float* biasIm,
                          global float2* derU, global float2* derV,
                          unsigned int outChannel)
{
    const int gsiz = get_global_size(0);
    const int n = get_global_id(0);
    const int m = get_global_id(1);
    const int inIdx = outChannel*inputOffset + m*gsiz+n;
    const int outIdx = outChannel*outputOffset + m*gsiz+n;


    float2 act = (float2)(actPotRe[inIdx]+biasRe[outChannel], actPotIm[inIdx]+biasIm[outChannel]);


    outRe[outIdx] = sinh(act.x)*cos(act.y);
    outIm[outIdx] = cosh(act.x)*sin(act.y);

    derU[outIdx].x = cosh(act.x)*cos(act.y);
    derU[outIdx].y = -sinh(act.x)*sin(act.y);
    derV[outIdx].x = -derU[outIdx].y;
    derV[outIdx].y = derU[outIdx].x;
}


__kernel void complexsinh_norm(global float* outRe, global float* outIm, const int outputOffset,
                          const global float* actPotRe, const global float* actPotIm, const int inputOffset,
                          const global float* biasRe, const global float* biasIm,
                          global float2* derU, global float2* derV,
                          unsigned int outChannel)
{
    const int gsiz = get_global_size(0);
    const int n = get_global_id(0);
    const int m = get_global_id(1);
    const int inIdx = outChannel*inputOffset + m*gsiz+n;
    const int outIdx = outChannel*outputOffset + m*gsiz+n;


    float2 act = (float2)(actPotRe[inIdx]+biasRe[outChannel], actPotIm[inIdx]+biasIm[outChannel]);


    outRe[outIdx] = sinh(act.x)*cos(act.y);
    outIm[outIdx] = cosh(act.x)*sin(act.y);

    derU[outIdx].x = cosh(act.x)*cos(act.y);
    derU[outIdx].y = -sinh(act.x)*sin(act.y);
    derV[outIdx].x = -derU[outIdx].y;
    derV[outIdx].y = derU[outIdx].x;
}
