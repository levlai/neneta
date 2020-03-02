
// In case of real act func delta_left = delta_right_re(actder_x + j actder_y)
// f_x= 2x, f_y = 2y
float2 absoluteDerImpl(float2 z)
{
    //return (float2)(2*z.x,2*z.y);
    return (float2)(z.x,z.y);
}

// f = x^2 + y^2
float2 absoluteImpl(float2 z)
{
    return (float2)(pow(z.x,2) + pow(z.y,2), 0);
}

__kernel void absolute(global float* inputRe, global float* inputIm, global float* actDerRe, global float* actDerIm)
{
    uint n = get_global_id(0);
    float2 derivation = absoluteDerImpl((float2)(inputRe[n], inputIm[n]));
    actDerRe[n] = derivation.x;
    actDerIm[n] = derivation.y;
    float2 absol = absoluteImpl((float2)(inputRe[n], inputIm[n]));
    inputRe[n] = absol.x;
    inputIm[n] = absol.y;
}

__kernel void calculateErrorsPL(const global float* weightsRe, const global float* weightsIm,
                           const global float* deltasRe, const global float* deltasIm,
                           global float* errorsRe, global float* errorsIm)
{
    uint leftNeuronId = get_global_id(0);

    float2 Wkm_c = (float2)(weightsRe[leftNeuronId], -weightsIm[leftNeuronId]);
    float2 deltak = (float2)(deltasRe[leftNeuronId], deltasIm[leftNeuronId]);
    float2 deltam = cmul(Wkm_c, deltak);

    errorsRe[leftNeuronId] = deltam.x;
    errorsIm[leftNeuronId] = deltam.y;

    //printf("NeuronId = %d -> Wkm_c = (%f, %f), deltam = (%f, %f) , error = (%f,%f) \n", leftNeuronId, Wkm_c.x, Wkm_c.y, deltam.x, deltam.y, errorsRe[leftNeuronId], errorsIm[leftNeuronId]);
}
