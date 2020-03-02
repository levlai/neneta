
__kernel void sumOfExponents(int size, global float* partialSumsRe, global const float* inputRe, local float* localSumsRe)
 {    
    uint global_id = get_global_id(0);

    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);

    // Copy from global to local memory
    if(global_id < size) //bounds checking
    {
        localSumsRe[local_id] = inputRe[global_id];
        //printf((__constant char *)"inputglobal %f\n", inputRe[global_id]);
    }

    // Loop for computing localSums : divide WorkGroup into 2 parts
    for(uint stride = group_size/2; stride>0; stride /=2)
    {
        // Waiting for each 2x2 addition into given workgroup
        barrier(CLK_LOCAL_MEM_FENCE);

        // Add elements 2 by 2 between local_id and local_id + stride
        if(local_id < stride && (global_id+stride) < size && global_id < size)
        {
            localSumsRe[local_id] += localSumsRe[local_id + stride];
        }
    }

    // Write result into partialSums[nWorkGroups]
    if (local_id == 0)
    {
        partialSumsRe[get_group_id(0)] = localSumsRe[0];

    }
}

__kernel void softmax(const global float* actPotRe, const global float* actPotIm,
                        global float* outputRe, global float* outputIm,
                        const global float* sumMod, const global float* maxElement)
 {
    uint global_id = get_global_id(0);

    float act = exp(actPotRe[global_id] - *maxElement - log(*sumMod));
   // printf("global_id = %d, softmax act = %f maxElement = %f, sum = %f, input = %f, log = %f\n", global_id, act, *maxElement, *sumMod, actPotRe[global_id], log(*sumMod));
    outputRe[global_id] = act;
    outputIm[global_id] = 0;

}

__kernel void addBias(global float* outRe, global float* outIm,
                      const global float* sumRe, const global float* sumIm,
                      const global float* biasRe, const global float* biasIm,
                      unsigned int outChannel)
{
    outRe[outChannel] = *sumRe+biasRe[outChannel];
    outIm[outChannel] = 0;
}
