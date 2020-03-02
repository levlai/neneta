
//A = A.*B
__kernel void calculateLocalGradient(global float* deltaRe, global float* deltaIm, const int deltaOffset,
                                     const global float* rightRe, const global float* rightIm, const int rightOffset,
                                     const int numOfKernels)
{
    const int width = get_global_size(0);
    const int m = get_global_id(1);
    const int n = get_global_id(0);
    const int idx = width*m + n;
    float2 der_conj = (float2)(0,0);
    float2 hp = (float2)(0,0);

    for(int channelId = 0; channelId < numOfKernels; ++channelId)
    {
        der_conj = (float2)(deltaRe[channelId*deltaOffset+idx], -deltaIm[channelId*deltaOffset+idx]);
        hp = cmul(der_conj, (float2)(rightRe[channelId*rightOffset+idx], rightIm[channelId*rightOffset+idx]));

        deltaRe[channelId*deltaOffset+idx] = hp.x;
        deltaIm[channelId*deltaOffset+idx] = hp.y;
    }

}

//flipping the matrix. If dim = 1 size = height, if dim = 2 size = width
__kernel void flipdim(global float* re, global float* im, const int channel, const int dim)
{
    const int xOut = get_global_id(0);
    const int yOut = get_global_id(1);
    const int size = max(get_global_size(0), get_global_size(1));
    const int offset = channel*size*size;

    float2 tmp = (float2)(re[offset + size*yOut+xOut], im[offset + size*yOut+xOut]);
    if(dim == 1)
    {
        re[offset + size*yOut+xOut] = re[offset + size*yOut+size-xOut-1];
        im[offset + size*yOut+xOut] = im[offset + size*yOut+size-xOut-1];
        re[offset + size*yOut+size-xOut-1] = tmp.x;
        im[offset + size*yOut+size-xOut-1] = tmp.y;
    }
    else
    {
        re[offset + size*yOut+xOut] = re[offset + size*(size-yOut-1)+xOut];
        im[offset + size*yOut+xOut] = im[offset + size*(size-yOut-1)+xOut];
        re[offset + size*(size-yOut-1)+xOut] = tmp.x;
        im[offset + size*(size-yOut-1)+xOut] = tmp.y;
    }
}

__kernel void compconv(const global float* reInput, const global float* imInput, const int offsetInput,
                       const global float* reKernel, const global float* imKernel, const int offsetKernel,
                       global float* reOutput, global float* imOutput, const int offsetOutput,
                       const int inputWidth, const int kernelWidth,
                       const int kernelId, const int numOfChannels, const int stride)
{
    const int convSize = get_global_size(0);
    const int xOut = get_global_id(0);
    const int yOut = get_global_id(1);

    const int xInTopLeft = stride*xOut;
    const int yInTopLeft = stride*yOut;


    float2 sum = (float2)(0,0);

    for(int channelId = 0; channelId < numOfChannels; ++channelId)
    {
        for(int r = 0; r < kernelWidth; ++r)
        {
            const int flipr = kernelWidth - 1 - r;
            const int idxKernelTmp = flipr*kernelWidth;
            const int yIn = yInTopLeft + r;
            const int idxInTmp =  yIn*inputWidth+xInTopLeft;

            for(int c = 0; c < kernelWidth; ++c)
            {
                const int flipc = kernelWidth - 1 - c;
                const int idxKernel = channelId*offsetKernel + idxKernelTmp + flipc; // kernels are stored in one array
                const int idxIn = channelId*offsetInput + idxInTmp + c; //as well as input channels
                sum = cadd(sum, cmul((float2)(reInput[idxIn], imInput[idxIn]), (float2)(reKernel[idxKernel],imKernel[idxKernel])));
            }
        }
    }

    const int idxOut = kernelId*offsetOutput + yOut * convSize + xOut;
    reOutput[idxOut] = sum.x;
    imOutput[idxOut] = sum.y;    
    //printf("output = (%f,%f) (%d,%d)\n", sum.x, sum.y, xOut, yOut);
}

__kernel void updateWeights(const global float* reInput, const global float* imInput, const int offsetInput,
                            const global float* reDeltas, const global float* imDeltas, const int offsetKernel,
                            global float* reWeights, global float* imWeights, const int offsetOutput,
                            global float* reBiases, global float* imBiases,
                            const int inputWidth, const int kernelWidth,
                            const int kernelId, const int numOfChannels, const int stride)
{
    const float eta = 0.01f;
    const float tsetsize = 50000.0f;
    const float lambda = 1.0f;

    const int corrSize = get_global_size(0);
    const int xOut = get_global_id(0);
    const int yOut = get_global_id(1);

    const int xInTopLeft = stride*xOut;
    const int yInTopLeft = stride*yOut;


    float2 sum;
    float2 biasSum;
    int idxOut;    
    for(int channelId = 0; channelId < numOfChannels; ++channelId)
    {
        sum = (float2)(0,0);
        biasSum = (float2)(0,0);
        for(int r = 0; r < kernelWidth; ++r)
        {
            const int idxKernelTmp = r*kernelWidth;
            const int yIn = yInTopLeft + r;
            const int idxInTmp =  yIn*inputWidth+xInTopLeft;

            for(int c = 0; c < kernelWidth; ++c)
            {
                const int idxKernel = kernelId*offsetKernel + idxKernelTmp + c;
                const int idxIn = channelId*offsetInput + idxInTmp + c;
                sum = cadd(sum, cmul((float2)(reInput[idxIn], -imInput[idxIn]), (float2)(reDeltas[idxKernel],imDeltas[idxKernel]))); //conjugated inputs
                biasSum = cadd(biasSum, cmul((float2)(1, 0), (float2)(reDeltas[idxKernel],imDeltas[idxKernel]))); //conjugated inputs
            }
        }        
        idxOut = channelId*offsetOutput + yOut * corrSize + xOut;
        //printf("weights update idxOut %d, delta = (%f, %f)\n", idxOut, sum.x, sum.y);
        reWeights[idxOut] = (1.0f-eta*lambda/tsetsize)*reWeights[idxOut] - eta*sum.x;
        imWeights[idxOut] = (1.0f-eta*lambda/tsetsize)*imWeights[idxOut] - eta*sum.y;
        reBiases[kernelId] -= eta*biasSum.x;
        imBiases[kernelId] -= eta*biasSum.y;

    }    
}

//padding left and top with outputSize-inputWidth
__kernel void calculateErrors(const global float* reInput, const global float* imInput, const int offsetInput, //deltas
                            const global float* reKernel, const global float* imKernel, const int offsetKernel,//weights
                            global float* reOutput, global float* imOutput,                                    //shmembkp
                            const int inputWidth, const int kernelWidth,
                            const int kernelId, const int numOfChannels, const int stride)
{

    const int corrSize = get_global_size(0);
    const int padding = corrSize - inputWidth;
    const int xOut = get_global_id(0);
    const int yOut = get_global_id(1);

    const int xInTopLeft = stride*xOut-padding;
    const int yInTopLeft = stride*yOut-padding;

    float2 sum = (float2)(0,0);
    int idxOut = 0;
    for(int channelId = 0; channelId < numOfChannels; ++channelId)
    {
        idxOut =  channelId*corrSize*corrSize + yOut * corrSize + xOut;
        sum = (float2)(0,0);
        if(kernelId != 0)
        {
            //only for a first kernel sum starts from 0
            sum = (float2)(reOutput[idxOut],  imOutput[idxOut]);
        }

        for(int r = 0; r < kernelWidth; ++r)
        {
            const int idxKernelTmp = r*kernelWidth;
            const int yIn = yInTopLeft + r;
            if(yIn >= 0 && yIn < inputWidth)
            {
                const int idxInTmp =  yIn*inputWidth+xInTopLeft;
                for(int c = 0; c < kernelWidth; ++c)
                {
                    if(xInTopLeft+c >= 0 && xInTopLeft+c < inputWidth)
                    {
                        const int idxKernel = channelId*offsetKernel + idxKernelTmp + c; // kernels are stored in one array
                        const int idxIn = kernelId*offsetInput + idxInTmp + c; //as well as input channels
                        sum = cadd(sum, cmul((float2)(reInput[idxIn],imInput[idxIn]), (float2)(reKernel[idxKernel], -imKernel[idxKernel]))); //conjugate weights
                    /*    if(idxOut == 49 || idxOut == 0)
                        {
                            printf("idxout = %d weights(%d)*input(%d) and adding to %f,%f, channel %d\n", idxOut, idxKernel, idxIn, sum.x, sum.y, channelId);
                            printf("in(%f,%f), knl(%f,%f)\n", reInput[idxIn], imInput[idxIn], reKernel[idxKernel], imKernel[idxKernel]);
                            printf("sum(%f,%f)\n", sum.x, sum.y);
                        } */
                    }
                }
            }
        }
        reOutput[idxOut] = sum.x;
        imOutput[idxOut] = sum.y;
    }
}
