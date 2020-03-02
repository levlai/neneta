
int reverse(unsigned short number, unsigned short numOfBits)
{
        unsigned short r, i;
        for (r = 0, i = 0; i < numOfBits; ++i)
        {
                r |= ((number >> i) & 1) << (numOfBits - i - 1);
        }
        return r;
}

__kernel void rowIndexReverse(global float* Re, global float* Im, int imgDim, int numOfBits)
{
    int m = get_global_id(1);    
    int reversedIndexM = reverse(m, numOfBits);        
    if(m < reversedIndexM)
    {
        int n = get_global_id(0);
        float temp = Re[m*imgDim + n];
        Re[m*imgDim + n] = Re[reversedIndexM*imgDim + n];
        Re[reversedIndexM*imgDim + n] = temp;

        temp = Im[m*imgDim + n];
        Im[m*imgDim + n] = Im[reversedIndexM*imgDim + n];
        Im[reversedIndexM*imgDim + n] = temp;
    }
}

__kernel void columnIndexReverse(global float* Re, global float* Im, int imgDim, int numOfBits)
{
    int n = get_global_id(0);
    int reversedIndexN = reverse(n, numOfBits);
    if(n < reversedIndexN)
    {
        int m = get_global_id(1);
        float temp = Re[m*imgDim + n];
        Re[m*imgDim + n] = Re[m*imgDim + reversedIndexN];
        Re[m*imgDim + reversedIndexN] = temp;

        temp = Im[m*imgDim + n];
        Im[m*imgDim + n] = Im[m*imgDim + reversedIndexN];
        Im[m*imgDim + reversedIndexN] = temp;
    }
}

__kernel void dit_2x2radix_2dfft_1st(global float* Re, global float* Im, global float* ReTmp, global float* ImTmp, int size, int stage, unsigned int reverse)
{
    float sign = 1.0f;
    if(reverse == 1) sign = -1.0f;
    unsigned char S;
    int N, Nhalf, m, n, k1, k1next, k2, k2next;
    float ReWnk, ImWnk, Arg;

    N = pown(2.0f, stage);
    Nhalf = N/2;

    S = 0;
    m = get_global_id(1);
    k1 = m;
    k1next = k1+Nhalf;
    n = get_global_id(0);
    k2 = n;
    k2next = k2+Nhalf;

    if(m%N >= Nhalf)
    {
        S |= 2;
        k1next = k1;
        k1 -= Nhalf;
    }

    if(n%N >= Nhalf)
    {
        S |= 1;
        k2next = k2;
        k2 -= Nhalf;
    }

    switch(S)
    {
        case 0: // S00
            ReTmp[k1*size + k2] = Re[k1*size + k2];
            ImTmp[k1*size + k2] = Im[k1*size + k2];
            Re[k1*size + k2] = 0;
            Im[k1*size + k2] = 0;
            break;
        case 1: // S01
            Arg = sign*2.0f*k2/N;
            ReWnk = cospi(Arg);
            ImWnk = -sinpi(Arg);
            ReTmp[k1*size + k2next] = Re[k1*size + k2next]*ReWnk - Im[k1*size + k2next]*ImWnk;
            ImTmp[k1*size + k2next] = Re[k1*size + k2next]*ImWnk + Im[k1*size + k2next]*ReWnk;
            Re[k1*size + k2next] = 0;
            Im[k1*size + k2next] = 0;
            break;
        case 2: // S10
            Arg = sign*2.0f*k1/N;
            ReWnk = cospi(Arg);
            ImWnk = -sinpi(Arg);
            ReTmp[k1next*size + k2] = Re[k1next*size + k2]*ReWnk - Im[k1next*size + k2]*ImWnk;
            ImTmp[k1next*size + k2] = Re[k1next*size + k2]*ImWnk + Im[k1next*size + k2]*ReWnk;
            Re[k1next*size + k2] = 0;
            Im[k1next*size + k2] = 0;
            break;
        case 3: // S11
            Arg = sign*2.0f*(k1+k2)/N;
            ReWnk = cospi(Arg);
            ImWnk = -sinpi(Arg);
            ReTmp[k1next*size + k2next] = Re[k1next*size + k2next]*ReWnk - Im[k1next*size + k2next]*ImWnk;
            ImTmp[k1next*size + k2next] = Re[k1next*size + k2next]*ImWnk + Im[k1next*size + k2next]*ReWnk;
            Re[k1next*size + k2next] = 0;
            Im[k1next*size + k2next] = 0;
            break;
    }
}

void float_atomic_add(volatile __global float* addr, float val)
{
   union
   {
       unsigned int u32;
       float        f32;
   } next, expected, current;

   current.f32 = *addr;
   do{
       expected.f32 = current.f32;
       next.f32     = expected.f32 + val;
            current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr,
                           expected.u32, next.u32);
   } while( current.u32 != expected.u32 );
}

__kernel void dit_2x2radix_2dfft_2nd(global float* Re, global float* Im, global float* ReTmp, global float* ImTmp, int size, int stage)
{
    unsigned char S;
    int N, Nhalf, m, n, k1, k1next, k2, k2next;
    float NewRe, NewIm;

    N = pow((float)2,(float)stage);
    Nhalf = N/2;

    S = 0;
    m = get_global_id(1);
    k1 = m;
    k1next = k1+Nhalf;
    n = get_global_id(0);
    k2 = n;
    k2next = k2+Nhalf;

    if(m%N >= Nhalf)
    {
        S |= 2;
        k1next = k1;
        k1 -= Nhalf;
    }

    if(n%N >= Nhalf)
    {
        S |= 1;
        k2next = k2;
        k2 -= Nhalf;
    }

    switch(S)
    {
        case 0: // S00
            NewRe = ReTmp[k1*size + k2];
            NewIm = ImTmp[k1*size + k2];
            float_atomic_add(&Re[k1*size + k2], NewRe);
            float_atomic_add(&Im[k1*size + k2], NewIm);
            float_atomic_add(&Re[k1*size + k2next], NewRe);
            float_atomic_add(&Im[k1*size + k2next], NewIm);
            float_atomic_add(&Re[k1next*size + k2], NewRe);
            float_atomic_add(&Im[k1next*size + k2], NewIm);
            float_atomic_add(&Re[k1next*size + k2next], NewRe);
            float_atomic_add(&Im[k1next*size + k2next], NewIm);
            break;
        case 1: // S01
            NewRe = ReTmp[k1*size + k2next];
            NewIm = ImTmp[k1*size + k2next];
            float_atomic_add(&Re[k1*size + k2], NewRe);
            float_atomic_add(&Im[k1*size + k2], NewIm);
            float_atomic_add(&Re[k1*size + k2next], -NewRe);
            float_atomic_add(&Im[k1*size + k2next], -NewIm);
            float_atomic_add(&Re[k1next*size + k2], NewRe);
            float_atomic_add(&Im[k1next*size + k2], NewIm);
            float_atomic_add(&Re[k1next*size + k2next], -NewRe);
            float_atomic_add(&Im[k1next*size + k2next], -NewIm);
            break;
        case 2: // S10
            NewRe = ReTmp[k1next*size + k2];
            NewIm = ImTmp[k1next*size + k2];
            float_atomic_add(&Re[k1*size + k2], NewRe);
            float_atomic_add(&Im[k1*size + k2], NewIm);
            float_atomic_add(&Re[k1*size + k2next], NewRe);
            float_atomic_add(&Im[k1*size + k2next], NewIm);
            float_atomic_add(&Re[k1next*size + k2], -NewRe);
            float_atomic_add(&Im[k1next*size + k2], -NewIm);
            float_atomic_add(&Re[k1next*size + k2next], -NewRe);
            float_atomic_add(&Im[k1next*size + k2next], -NewIm);
            break;
        case 3: // S11
            NewRe = ReTmp[k1next*size + k2next];
            NewIm = ImTmp[k1next*size + k2next];
            float_atomic_add(&Re[k1*size + k2], NewRe);
            float_atomic_add(&Im[k1*size + k2], NewIm);
            float_atomic_add(&Re[k1*size + k2next], -NewRe);
            float_atomic_add(&Im[k1*size + k2next], -NewIm);
            float_atomic_add(&Re[k1next*size + k2], -NewRe);
            float_atomic_add(&Im[k1next*size + k2], -NewIm);
            float_atomic_add(&Re[k1next*size + k2next], NewRe);
            float_atomic_add(&Im[k1next*size + k2next], NewIm);
            break;
    }
}

__kernel void print(global float* Re, global float* Im, global float* ReTmp, global float* ImTmp, int size, int stage)
{
    int m = get_global_id(1);
    int n = get_global_id(0);

    //printf("stage %d, m,n = %d,%d, Re = %f Im = %f - ReTmp = %f ImTmp = %f-> S00\n", stage, m, n, Re[m*size + n], Im[m*size + n],  ReTmp[m*size + n], ImTmp[m*size + n]);
}

__kernel void fftShift(global float* Re, global float* Im, int imgDim)
{
    int n = get_global_id(0);
    int m = get_global_id(1); //must be smaller than imgDim/2
    int halfImgDim = imgDim/2;
    if(n < halfImgDim) // first quadrant swaps with fourth
    {
        int mswap = m + halfImgDim;
        int nswap = n + halfImgDim;

        float tmp = Re[m*imgDim + n];
        Re[m*imgDim + n] = Re[mswap*imgDim + nswap];
        Re[mswap*imgDim + nswap] = tmp;

        tmp = Im[m*imgDim + n];
        Im[m*imgDim + n] = Im[mswap*imgDim + nswap];
        Im[mswap*imgDim + nswap] = tmp;
    }
    else               // second quadrant swaps with third
    {
        int mswap = m + halfImgDim;
        int nswap = n - halfImgDim;

        float tmp = Re[m*imgDim + n];
        Re[m*imgDim + n] = Re[mswap*imgDim + nswap];
        Re[mswap*imgDim + nswap] = tmp;

        tmp = Im[m*imgDim + n];
        Im[m*imgDim + n] = Im[mswap*imgDim + nswap];
        Im[mswap*imgDim + nswap] = tmp;
    }
}


__kernel void fftScale(global float* Re, global float* Im, int imgDim)
{
    int n = get_global_id(0);
    int m = get_global_id(1);

    float scalingFactor = 1.0f/(float)(imgDim*imgDim);
    Re[m*imgDim + n] = scalingFactor*Re[m*imgDim + n];
    Im[m*imgDim + n] = scalingFactor*Im[m*imgDim + n];
}
