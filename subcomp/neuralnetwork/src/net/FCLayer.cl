__kernel void sum(int size, global float* partialSumsRe, global float* partialSumsIm,
                  global const float* inputRe, global const float* inputIm,
                  local float* localSumsRe, local float* localSumsIm)
 {    
    uint global_id = get_global_id(0);

    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);

    // Copy from global to local memory
    if(global_id < size) //bounds checking
    {
        localSumsRe[local_id] = inputRe[global_id];
        localSumsIm[local_id] = inputIm[global_id];
    }
    else
    {
        localSumsRe[local_id] = 0;
        localSumsIm[local_id] = 0;
    }

    // Loop for computing localSums : divide WorkGroup into 2 parts
    for(uint stride = group_size/2; stride>0; stride /=2)
    {
        // Waiting for each 2x2 addition into given workgroup
        barrier(CLK_LOCAL_MEM_FENCE);

        // Add elements 2 by 2 between local_id and local_id + stride
        if(local_id < stride)
        {
            localSumsRe[local_id] += localSumsRe[local_id + stride];
            localSumsIm[local_id] += localSumsIm[local_id + stride];            
        }
    }

    // Write result into partialSums[nWorkGroups]
    if (local_id == 0)
    {
        partialSumsRe[get_group_id(0)] = localSumsRe[0];
        partialSumsIm[get_group_id(0)] = localSumsIm[0];        
       //printf((__constant char *)"size %d sums %f \n", size, partialSumsRe[get_group_id(0)]);
    }

}

__kernel void println(global char* string, int length)
{
    printf((__constant char *)"%s", "\nkernel msg:\n");
    for(int i = 0; i < length; ++i)
    {
        printf((__constant char *)"%c", string[i]);
    }
    printf((__constant char *)"%s", "\n");
}


