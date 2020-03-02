
__kernel void updateWeightsFC(global float* weightsRe, global float* weightsIm,
                              const global float2* deltasRe, const global float2* deltasIm,
                              global float* biasesRe, global float* biasesIm,
                              const global float* inputRe, const global float* inputIm,
                              const unsigned int neuronId)
{
    const float eta = 0.0005f;
    const float tsetsize = 50000.0f;
    const float lambda = 5.0f;

    unsigned int inputNeuronId = get_global_id(0);

    float2 localGradient = (float2)(deltasRe[neuronId].x, deltasIm[neuronId].x);
    float2 Xmk_c = (float2)(inputRe[inputNeuronId], -inputIm[inputNeuronId]);
    float2 DWmk = eta*cmul(Xmk_c, localGradient);    

    if(inputNeuronId == 0)
    {
        float2 DBias = eta*cmul((float2)(1,0), localGradient);
        biasesRe[neuronId] -= DBias.x;
        biasesIm[neuronId] -= DBias.y;

      //  printf("NeuronId = %d; localGradient = (%f, %f)\n", neuronId, localGradient.x, localGradient.y);
    }

    //printf("DWmk_%d = (%f, %f)\n", inputNeuronId, cmul(Xmk_c, localGradient).x, cmul(Xmk_c, localGradient).y);
    weightsRe[inputNeuronId] = ((1.0f-eta*lambda/tsetsize)*weightsRe[inputNeuronId] - DWmk.x);
    weightsIm[inputNeuronId] = ((1.0f-eta*lambda/tsetsize)*weightsIm[inputNeuronId] - DWmk.y);
}

__kernel void calculateErrorsFC(const global float* weightsRe, const global float* weightsIm,
                           const global float2* deltasRe, const global float2* deltasIm,
                           global float* errorsRe, global float* errorsIm,
                           const int neuronId)
{
    uint leftNeuronId = get_global_id(0);

    float2 Wkm_c = (float2)(weightsRe[leftNeuronId], -weightsIm[leftNeuronId]);    
    float2 deltak = (float2)(deltasRe[neuronId].x, deltasIm[neuronId].x);
    float2 deltam = cmul(Wkm_c, deltak);    
    if(neuronId != 0)
    {
        errorsRe[leftNeuronId] += deltam.x;
        errorsIm[leftNeuronId] += deltam.y;
    }
    else
    {
        errorsRe[leftNeuronId] = deltam.x;
        errorsIm[leftNeuronId] = deltam.y;        
    }
    //printf("FC NeuronId = %d -> Wkm_c = (%f, %f), deltam = (%f, %f), error = (%f,%f) \n", neuronId, Wkm_c.x, Wkm_c.y, deltam.x, deltam.y, errorsRe[leftNeuronId], errorsIm[leftNeuronId]);
}

__kernel void calculateDeltasFC(const global float* errorsRe, const global float* errorsIm,
                                global float2* deltasU, global float2* deltasV)
{
    uint currentNeuron = get_global_id(0);

  //  float2 der_conj = (float2)(deltasRe[currentNeuron], -deltasIm[currentNeuron]);
  //  float2 hp = cmul(der_conj, (float2)(errorsRe[currentNeuron], errorsIm[currentNeuron]));

    float2 delta = errorsRe[currentNeuron]*deltasU[currentNeuron] + errorsIm[currentNeuron]*deltasV[currentNeuron];

  //  printf("inid %d err = (%f,%f) u=(%f, %f) v=(%f, %f) localGradient = (%f, %f)\n", currentNeuron, errorsRe[currentNeuron], errorsIm[currentNeuron],
  //  deltasU[currentNeuron].x, deltasU[currentNeuron].y, deltasV[currentNeuron].x, deltasV[currentNeuron].y, delta.x, delta.y);

    deltasU[currentNeuron].x = delta.x;
    deltasU[currentNeuron].y = 0;
    deltasV[currentNeuron].x = delta.y;
    deltasV[currentNeuron].y = 0;
}

__kernel void swapbuffers(global float* reA, global float* reB, global float* imA, global float* imB, uint size)
{
    uint m = get_global_id(0);
    if(m < size)
    {
        float tmp = reA[m];
        reA[m] = reB[m];
        reB[m] = tmp;

        tmp = imA[m];
        imA[m] = imB[m];
        imB[m] = tmp;
    }
}

