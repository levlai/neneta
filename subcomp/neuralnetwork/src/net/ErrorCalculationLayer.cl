
// error = desired - output;
__kernel void meansquare(global float* errRe, global float* errIm,
                        const global float* desRe, const global float* desIm,
                        const global float* outRe, const global float* outIm)
{
    uint n = get_global_id(0);
    errRe[n] = desRe[n] - outRe[n];
    errIm[n] = desIm[n] - outIm[n];
}

__kernel void complexms(global float* errRe, global float* errIm,
                        const global float* desRe, const global float* desIm,
                        const global float* outRe, const global float* outIm)
{
    uint n = get_global_id(0);
    errRe[n] = -(desRe[n] - outRe[n]);
    errIm[n] = -(desIm[n] - outIm[n]);
  //  printf("n=%d error = (%f, %f) \n", n, errRe[n], errIm[n]);
}

__kernel void complexms_act(global float* loss,
                            const global float* desRe, const global float* desIm,
                            const global float* outRe, const global float* outIm)
{
    int n = get_global_id(0);
    float2 tmp = cmul((float2)(desRe[n] - outRe[n], desIm[n] - outIm[n]), (float2)(desRe[n] - outRe[n], -(desIm[n] - outIm[n])));
    //printf("n=%d des = (%f, %f), out = (%f, %f) \n", n, desRe[n], desIm[n], outRe[n], outIm[n]);
    loss[n] = 0.5*tmp.x;    
}

__kernel void complexms_acc(const global float* desRe, const global float* desIm,
                           const global float* outRe, const global float* outIm,
                           unsigned int size, global float* accuracy)
{
    *accuracy = 0.0f;
    float calculatedDistance = 0.0f;
    int id = 0;
    for(int i = 0; i < size; ++i)
    {
        if(desRe[i]==1)
        {
            id = i;
            calculatedDistance = cabs((float2)(desRe[i], desIm[i]) - (float2)(outRe[i], outIm[i]));
        }
    }

    for(int i = 0; i < size; ++i)
    {
        if(id != i && cabs((float2)(desRe[id], desIm[id]) - (float2)(outRe[i], outIm[i])) < calculatedDistance)
        {
            return;
        }
    }
    *accuracy = 1.0f;
}

__kernel void crossentropy(global float* errRe, global float* errIm,
                           const global float* desRe, const global float* desIm,
                           const global float* outRe, const global float* outIm)
{
    uint n = get_global_id(0);
    errRe[n] = outRe[n]-desRe[n];
    errIm[n] = outIm[n]-desIm[n];
    //printf("cross entropy error = (%f, %f) out = (%f, %f), des = (%f, %f), n = %d \n", errRe[n], errIm[n], outRe[n], outIm[n], desRe[n], desIm[n], n);
}

__kernel void crossentropy_acc(const global float* desRe, const global float* desIm,
                           const global float* outRe, const global float* outIm,
                           unsigned int size, global float* accuracy)
{
    *accuracy = 0.0f;
    float calculated = 0.0f;
    for(int i = 0; i < size; ++i)
    {
        if(desRe[i]==1)
        {
            calculated = outRe[i];
        }
    }

    for(int i = 0; i < size; ++i)
    {
        if(calculated < outRe[i])
        {
            return;
        }
    }
    *accuracy = 1.0f;
}

__kernel void loss_sum(global float* partialSum,
                               global const float* input,
                               const int size, const int vecSize,
                               local float* localSum)
 {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);

    uint group_size = get_local_size(0);

    // Copy from global to local memory
    if(global_id < size) //bounds checking
    {
        localSum[local_id] = input[global_id];
    }

    // Loop for computing localSums : divide WorkGroup into 2 parts
    for(uint stride = group_size/2; stride>0; stride /=2)
    {
        // Waiting for each 2x2 addition into given workgroup
        barrier(CLK_LOCAL_MEM_FENCE);

        // Add elements 2 by 2 between local_id and local_id + stride
        if(local_id < stride && (global_id+stride) < size && global_id < size)
        {
            localSum[local_id] += localSum[local_id + stride];
        }
    }

    if (local_id == 0)
    {
        partialSum[get_group_id(0)] = localSum[0];
     //   printf("error %f \n", partialSum[get_group_id(0)]);
    }
}

__kernel void crossentropy_act(global float* loss,
                               const global float* desRe, const global float* desIm,
                               const global float* outRe, const global float* outIm)
{
    int n = get_global_id(0);
    //error[n] = desired[n]*log(output[n]) + (1-desired[n])*log(1.0f - output[n]);
    loss[n] = -desRe[n]*log(outRe[n]+0.000001f);
    //printf("desired=%f output=%f error=%f\n", desired[n], output[n], error[n]);
}
