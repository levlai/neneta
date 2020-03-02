//A = B.*C or A += B.*C
__kernel void hadamard_product(global float* ReA, global float* ImA,
                               global float* ReB, global float* ImB,
                               global float* ReC, global float* ImC, int imgDim, int ch)
{
    int m = get_global_id(1);
    int n = get_global_id(0);

    if(ch == 0)
    {
        ReA[m*imgDim+n] = ReC[m*imgDim+n]*ReB[m*imgDim+n] - ImC[m*imgDim+n]*ImB[m*imgDim+n];
        ImA[m*imgDim+n] = ReC[m*imgDim+n]*ImB[m*imgDim+n] + ImC[m*imgDim+n]*ReB[m*imgDim+n];
    }
    else
    {
        ReA[m*imgDim+n] += ReC[m*imgDim+n]*ReB[m*imgDim+n] - ImC[m*imgDim+n]*ImB[m*imgDim+n];
        ImA[m*imgDim+n] += ReC[m*imgDim+n]*ImB[m*imgDim+n] + ImC[m*imgDim+n]*ReB[m*imgDim+n];
    }
}


//A = B.*C
__kernel void vec_hadamard_product(global float* ReA, global float* ImA,
                                    global float* ReB, global float* ImB,
                                    global float* ReC, global float* ImC) //weights
{
    int n = get_global_id(0);    
    ReA[n] = ReC[n]*ReB[n] - ImC[n]*ImB[n];
    ImA[n] = ReC[n]*ImB[n] + ImC[n]*ReB[n];
   // printf("weight_%d = (%f,%f) Input = (%f,%f) Prod=(%f,%f)\n", n, ReC[n], ImC[n], ReB[n], ImB[n], ReA[n], ImA[n]);
}

__kernel void exponent(global float* inOut, const global float* scalar)
{
    inOut[get_global_id(0)] = exp(inOut[get_global_id(0)] - *scalar);
   // printf("n = %d, subInPlaceWithReal = %f\n", get_global_id(0), inOut[get_global_id(0)]);
}


// A = B[...[]...]
__kernel void spectral_pooling(global float* ReA, global float* ImA, global float* ReB, global float* ImB, int newSize, int oldSize)
{
    int newCenter = newSize/2;
    int q1m = get_global_id(1);
    int q1n = get_global_id(0);

    int oldq1m = q1m;
    int oldq1n = q1n;

    int oldq2m = q1m;
    int oldq2n = oldSize-2*newCenter+q1n;

    int oldq3m = oldSize-2*newCenter+q1m;
    int oldq3n = oldSize-2*newCenter+q1n;

    int oldq4m = oldSize-2*newCenter+q1m;
    int oldq4n = q1n;

    float norm = (float)(newSize*newSize)/(float)(oldSize*oldSize);

    if(q1m < newCenter && q1n < newCenter)
    {
        ReA[q1m*newSize+q1n] = norm*ReB[oldq1m*oldSize+oldq1n];
        ImA[q1m*newSize+q1n] = norm*ImB[oldq1m*oldSize+oldq1n];
    }
    else if(q1m < newCenter && q1n >= newCenter)
    {
        ReA[q1m*newSize+q1n] = norm*ReB[oldq2m*oldSize+oldq2n];
        ImA[q1m*newSize+q1n] = norm*ImB[oldq2m*oldSize+oldq2n];
    }
    else if(q1m >= newCenter && q1n >= newCenter)
    {
        ReA[q1m*newSize+q1n] = norm*ReB[oldq3m*oldSize+oldq3n];
        ImA[q1m*newSize+q1n] = norm*ImB[oldq3m*oldSize+oldq3n];
    } else if(q1m >= newCenter && q1n < newCenter)
    {
        ReA[q1m*newSize+q1n] = norm*ReB[oldq4m*oldSize+oldq4n];
        ImA[q1m*newSize+q1n] = norm*ImB[oldq4m*oldSize+oldq4n];
    }
}

__kernel void printMatrix(const global float* mat, const uint width, const uint height)
{
    for(int row = 0; row < height; ++row)
    {
        printf((__constant char *)"row %d: ", row);
        for(int col = 0; col < width; ++col)
        {
            printf((__constant char *)"%f ", mat[row*width+col]);
        }
        printf((__constant char *)"\t\n");
    }
}

__kernel
void maxValue(global float* buffer,
              local float* scratch,
              const int length,
              global float* result)
{
    int global_index = get_global_id(0);
    int local_index = get_local_id(0);

    // Load data into local memory
    if (global_index < length)
    {        
        scratch[local_index] = buffer[global_index];        
    }
    else
    {
        // Infinity is the identity element for the max operation
        scratch[local_index] = -INFINITY;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int offset = 1; offset < get_local_size(0); offset <<= 1) // 1, 2, 4, 8
    {
        int mask = (offset << 1) - 1; //0001b (0,2,4,6,8,10,12,14), 0011b (0,4,8,12) , 0111b (0,8)
        if ((local_index & mask) == 0)
        {
            float other = scratch[local_index + offset];
            float mine = scratch[local_index];
            scratch[local_index] = (mine > other) ? mine : other;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0)
    {
        result[get_group_id(0)] = scratch[0];        
    }
}
