#include "Utils.h"
#include <random>
#include <iostream>

namespace neneta
{
namespace net
{

std::vector<cmn::GPUFLOAT> generateRandomVector(unsigned int size, cmn::GPUFLOAT mean, cmn::GPUFLOAT deviation)
{
    static const float scale = 1.0f/std::sqrt(size);
    std::vector<cmn::GPUFLOAT> vec;    
    static std::default_random_engine generator;
    std::normal_distribution<cmn::GPUFLOAT> distribution(mean, deviation);
    auto rn = std::bind(distribution, std::ref(generator));
    for(unsigned int i = 0; i < size; ++i)
    {
        vec.push_back(scale*rn());
    }
    return vec;
}

gpu::OpenCLKernelParameters calculateKernelParameters(unsigned int inputSize, unsigned int maxWGSize)
{
    return gpu::OpenCLKernelParameters((inputSize <= maxWGSize? maxWGSize
                                     : (inputSize%maxWGSize) ? maxWGSize*(1+inputSize/maxWGSize)
                                                               : maxWGSize*(inputSize/maxWGSize))
            , maxWGSize);
}

void updateWeightsFC(gpu::OpenCLExecutionPlan& explan, LayerWeights& weights, const LayerDeltas& deltas,
                     const LayerBiases& layerBiases, const LayerInput& input, unsigned int numOfNeurons)
{
    assert(weights.size() > 0 && "no weights to update");
    for(cl_int neuronId = 0; neuronId < numOfNeurons; ++neuronId)
    {
        gpu::OpenCLKernelParameters kernelParams(weights[neuronId].m_size, 0);
        explan.planBck("updateWeightsFC", kernelParams, weights[neuronId].m_re, weights[neuronId].m_im,
                                                        deltas.m_re, deltas.m_im,
                                                        layerBiases.m_re, layerBiases.m_im,
                                                        input.m_re, input.m_im, neuronId);
    }
}

void swapBuffers(gpu::OpenCLExecutionPlan& explan, bool forward, unsigned int size, unsigned int maxwgsize, gpu::BufferIO sharedBuffer)
{
    gpu::OpenCLKernelParameters kernelParams = net::calculateKernelParameters(size, maxwgsize);
    if(forward)
    {
        explan.planFwd("swapbuffers", kernelParams, sharedBuffer.m_reShMem, sharedBuffer.m_reShMemBkp, sharedBuffer.m_imShMem, sharedBuffer.m_imShMemBkp, size);
    }
    else
    {
        explan.planBck("swapbuffers", kernelParams, sharedBuffer.m_reShMem, sharedBuffer.m_reShMemBkp, sharedBuffer.m_imShMem, sharedBuffer.m_imShMemBkp, size);
    }

}

void calculateErrorsFC(gpu::OpenCLExecutionPlan& explan, gpu::BufferIO sharedBuffer, const LayerWeights& weights, const LayerDeltas& deltas,
                  const unsigned int numOfNeuronsRight, const unsigned int numOfNeuronsLeft)
{
    assert(weights.size() > 0 && "missing weights");
    gpu::OpenCLKernelParameters kernelParams(numOfNeuronsLeft, 0);
    for(cl_int neuronId = 0; neuronId < numOfNeuronsRight; ++neuronId)
    {
        explan.planBck("calculateErrorsFC", kernelParams, weights[neuronId].m_re, weights[neuronId].m_im,
                                                          deltas.m_re, deltas.m_im,
                                                          sharedBuffer.m_reShMem, sharedBuffer.m_imShMem,
                                                          neuronId);
    }
}

void calculateErrorsPL(gpu::OpenCLExecutionPlan& explan, gpu::BufferIO sharedBuffer, const LayerWeights& weights, const LayerDeltas& deltas,
                       const unsigned int numOfNeuronsRight, const unsigned int numOfNeuronsLeft)
{
    //In projection layer number of neuron inputs is equal to number of neuron outputs
    assert(weights.size() > 0 && "missing weights");
    gpu::OpenCLKernelParameters kernelParams(numOfNeuronsLeft, 0);
    for(cl_int neuronId = 0; neuronId < numOfNeuronsRight; ++neuronId)
    {
        explan.planBck("calculateErrorsPL", kernelParams, weights[neuronId].m_re, weights[neuronId].m_im,
                                                          deltas.m_re, deltas.m_im,
                                                          sharedBuffer.m_reShMem, sharedBuffer.m_imShMem);
    }
}

void calculateDeltasFC(gpu::OpenCLExecutionPlan& explan, gpu::BufferIO sharedBuffer, const LayerDeltas& deltas)
{
    gpu::OpenCLKernelParameters kernelParams(deltas.m_size, 0);
    explan.planBck("calculateDeltasFC", kernelParams, sharedBuffer.m_reShMem, sharedBuffer.m_imShMem,
                                                      deltas.m_re, deltas.m_im);
}

void dbgPrintBuffer(gpu::OpenCLExecutionPlan& explan, const cl::CommandQueue& cq, cl::Buffer buffer, size_t size, const std::string& text)
{
    std::vector<cmn::GPUFLOAT> buff(size, 0);
    explan.readFromBuffer(cq, buffer, size*sizeof(cmn::GPUFLOAT), buff.data());
    std::cout << text << std::endl;
    int i = 0;
    for(const auto& el : buff)
    {
        std::cout << "\t[" << i++ << "] - " << el << std::endl;
    }
}

template<typename T>
void initBuffers(gpu::OpenCLExecutionPlan& explan, const cl::CommandQueue& cq, unsigned int size, cl::Buffer re, T valRe, cl::Buffer im, T valIm)
{    
    std::vector<T> reals(size, valRe);
    std::vector<T> imags(size, valIm);
    explan.writeToBuffer(cq, re, size*sizeof(T), reals.data());
    explan.writeToBuffer(cq, im, size*sizeof(T), imags.data());
}

void gpuPrintMatrix(gpu::OpenCLExecutionPlan& explan, const cl::CommandQueue& cq, const cl::Buffer& mat, const unsigned int width, const unsigned int height)
{
    gpu::OpenCLKernelParameters kernelParams(1, 1);
    explan.planFwd("printMatrix", kernelParams, mat, width, height);
}

void gpuPrintLine(gpu::OpenCLExecutionPlan& explan, const cl::CommandQueue& cq, std::string ln)
{
    static cl::Buffer line;
    static std::string tmp;
    tmp = ln;
    explan.writeToBuffer(cq, line, tmp.size()*sizeof(unsigned char), const_cast<char*>(tmp.c_str()));
    gpu::OpenCLKernelParameters kernelParams(1, 1);
    explan.planFwd("println", kernelParams, line, tmp.size());
}

void maxValue(gpu::OpenCLExecutionPlan& explan,
              const cl::Buffer& vec,
              cl::Buffer& res,
              cl::Buffer& tempRes,
              unsigned int vecSize,
              const unsigned int maxWGSize)
{
    do
    {
        gpu::OpenCLKernelParameters kparams = calculateKernelParameters(vecSize, maxWGSize);
        explan.planFwd("maxValue", kparams,
                vec,
                cl::Local(sizeof(cmn::GPUFLOAT)*maxWGSize),
                static_cast<cl_int>(vecSize),
                tempRes);

        vecSize = vecSize/maxWGSize + (vecSize%maxWGSize != 0);

        if(vecSize > 1)
        {
            explan.planCopyBufferFwd(tempRes, vec, vecSize*sizeof(cmn::GPUFLOAT));
        }
        else
        {
            explan.planCopyBufferFwd(tempRes, res, vecSize*sizeof(cmn::GPUFLOAT));
        }
    } while(vecSize > 1);
}

template
void initBuffers<cmn::GPUFLOAT2>(gpu::OpenCLExecutionPlan& explan, const cl::CommandQueue& cq, unsigned int size, cl::Buffer re, cmn::GPUFLOAT2 rv, cl::Buffer im, cmn::GPUFLOAT2 iv);

template
void initBuffers<cmn::GPUFLOAT>(gpu::OpenCLExecutionPlan& explan, const cl::CommandQueue& cq, unsigned int size, cl::Buffer re, cmn::GPUFLOAT rv, cl::Buffer im, cmn::GPUFLOAT iv);

}
}
