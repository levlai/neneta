#pragma once

#include <vector>
#include <OpenCLKernel.h>
#include <IOpenCLChainableExecutionPlan.h>
#include <OpenCLExecutionPlan.h>
#include <Types.h>

namespace neneta
{
namespace net
{

std::vector<cmn::GPUFLOAT> generateRandomVector(unsigned int size, cmn::GPUFLOAT mean, cmn::GPUFLOAT deviation);
gpu::OpenCLKernelParameters calculateKernelParameters(unsigned int size, unsigned int maxWGSize);

struct NeuronWeights
{
    NeuronWeights(const cl::Buffer& re, const cl::Buffer& im, unsigned int size) : m_re(re), m_im(im), m_size(size) {}
    cl::Buffer m_re;
    cl::Buffer m_im;
    unsigned int m_size;
};

//store for each layer
typedef std::vector<NeuronWeights> LayerWeights;
typedef NeuronWeights LayerInput;
typedef std::vector<NeuronWeights> ConvLayerDeltas;
typedef NeuronWeights LayerDeltas;
typedef NeuronWeights LayerBiases;

void updateWeightsFC(gpu::OpenCLExecutionPlan& explan, LayerWeights& weights, const LayerDeltas& deltas,
                     const LayerBiases& layerBiases, const LayerInput& input, unsigned int numOfNeurons);
void swapBuffers(gpu::OpenCLExecutionPlan& explan, bool forward, unsigned int size, unsigned int maxwgsize, gpu::BufferIO sharedBuffer);

void calculateErrorsFC(gpu::OpenCLExecutionPlan& explan, gpu::BufferIO sharedBuffer, const LayerWeights& weights, const LayerDeltas& deltas,
                       const unsigned int numOfNeuronsRight, const unsigned int numOfNeuronsLeft);
void calculateDeltasFC(gpu::OpenCLExecutionPlan& explan, gpu::BufferIO sharedBuffer, const LayerDeltas& deltas);
void calculateErrorsPL(gpu::OpenCLExecutionPlan& explan, gpu::BufferIO sharedBuffer, const LayerWeights& weights, const LayerDeltas& deltas,
                       const unsigned int numOfNeuronsRight, const unsigned int numOfNeuronsLeft);
void dbgPrintBuffer(gpu::OpenCLExecutionPlan& explan, const cl::CommandQueue& cq, cl::Buffer buffer, size_t size, const std::string& text);

template<typename T>
void initBuffers(gpu::OpenCLExecutionPlan& explan, const cl::CommandQueue& cq, unsigned int size, cl::Buffer re, T valRe, cl::Buffer im, T valIm);

void gpuPrintMatrix(gpu::OpenCLExecutionPlan& explan, const cl::CommandQueue& cq, const cl::Buffer& mat, const unsigned int width, const unsigned int height);
void gpuPrintLine(gpu::OpenCLExecutionPlan& explan, const cl::CommandQueue& cq, std::string ln);
void maxValue(gpu::OpenCLExecutionPlan& explan, const cl::Buffer& vec, cl::Buffer& res, cl::Buffer& tempRes, unsigned int vecSize, const unsigned int maxWGSize);
}
}
