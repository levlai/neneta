#include <ConvLayer.h>
#include <OpenCLContext.h>
#include <InputLayer.h>
#include "Utils.h"

using namespace neneta;
using namespace neneta::net;

ConvLayer::ConvLayer(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext)
    : gpu::OpenCLExecutionPlan(layerId, confReader, program)
    , m_persistance(confReader)
    , m_layerParameters(conf::ConvLayerConfiguration(getId(), confReader).getParamSet(getId()))
    , m_oclKernelParameters(m_layerParameters.m_inputSize, m_layerParameters.m_inputSize, getKernelConfiguration().getLWS(1), getKernelConfiguration().getLWS(2))
    , m_inputChannelSize(std::pow(m_layerParameters.m_inputSize, m_layerParameters.m_inputDim))
    , m_outputWidth((m_layerParameters.m_inputSize - m_layerParameters.m_kernelSize)/m_layerParameters.m_stride + 1)
    , m_outputChannelSize(std::pow(m_outputWidth, m_layerParameters.m_inputDim))
    , m_clContext(oclContext)
    , m_clProgram(program)
    , m_maxWGSize(m_clContext.getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>())
    , m_layerInput(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*m_inputChannelSize*sizeof(cmn::GPUFLOAT)),
                   cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*m_inputChannelSize*sizeof(cmn::GPUFLOAT)),
                   m_layerParameters.m_channels*m_inputChannelSize)
    , m_biases(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_numOfKernels*sizeof(cmn::GPUFLOAT)),
               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_numOfKernels*sizeof(cmn::GPUFLOAT)),
               m_layerParameters.m_numOfKernels)
    , m_deltas(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_numOfKernels*m_outputChannelSize*sizeof(cmn::GPUFLOAT2)),
               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_numOfKernels*m_outputChannelSize*sizeof(cmn::GPUFLOAT2)),
               m_layerParameters.m_numOfKernels*m_outputChannelSize)
    , m_weightsSizePerChannel(std::pow(m_layerParameters.m_kernelSize, m_layerParameters.m_inputDim))
{
    restore();
}

ConvLayer::ConvLayer(const conf::ConvLayerParams& params, const std::vector<cmn::GPUFLOAT>& weights, const std::vector<cmn::GPUFLOAT>& bias,
                     const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext)
    : gpu::OpenCLExecutionPlan(params.m_id, confReader, program)
    , m_persistance(confReader)
    , m_layerParameters(params)
    , m_oclKernelParameters(m_layerParameters.m_inputSize, m_layerParameters.m_inputSize, getKernelConfiguration().getLWS(1), getKernelConfiguration().getLWS(2))
    , m_inputChannelSize(std::pow(m_layerParameters.m_inputSize, m_layerParameters.m_inputDim))
    , m_outputWidth((m_layerParameters.m_inputSize - m_layerParameters.m_kernelSize)/m_layerParameters.m_stride + 1)
    , m_outputChannelSize(std::pow(m_outputWidth, m_layerParameters.m_inputDim))
    , m_clContext(oclContext)
    , m_clProgram(program)
    , m_maxWGSize(m_clContext.getDevice().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>())
    , m_layerInput(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*m_inputChannelSize*sizeof(cmn::GPUFLOAT)),
                   cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_channels*m_inputChannelSize*sizeof(cmn::GPUFLOAT)),
                   m_layerParameters.m_channels*m_inputChannelSize)
    , m_biases(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_numOfKernels*sizeof(cmn::GPUFLOAT)),
               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_numOfKernels*sizeof(cmn::GPUFLOAT)),
               m_layerParameters.m_numOfKernels)
    , m_deltas(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_numOfKernels*m_outputChannelSize*sizeof(cmn::GPUFLOAT2)),
               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_numOfKernels*m_outputChannelSize*sizeof(cmn::GPUFLOAT2)),
               m_layerParameters.m_numOfKernels*m_outputChannelSize)
    , m_weightsSizePerChannel(std::pow(m_layerParameters.m_kernelSize, m_layerParameters.m_inputDim))
{
    std::vector<cmn::GPUFLOAT> w = weights;
    w.insert(w.end(), bias.begin(), bias.end());
    init(w);
}

ConvLayer::~ConvLayer()
{
}

void ConvLayer::store()
{
    std::vector<cmn::GPUFLOAT> weightsandbiases(2*m_layerParameters.m_numOfKernels*(m_layerParameters.m_channels*m_weightsSizePerChannel+1));
    readWeightsAndBiases(weightsandbiases);
    m_persistance.storeFloatBlob(getId(), weightsandbiases);
}

void ConvLayer::readWeightsAndBiases(std::vector<cmn::GPUFLOAT>& weightsandbiases)
{
    unsigned int offset = 0;
    for(unsigned int i = 0; i < m_layerParameters.m_numOfKernels; ++i)
    {
        readFromBuffer(m_clContext.getCommandQueue(), m_weights[i].m_re, sizeof(cmn::GPUFLOAT)*m_layerParameters.m_channels*m_weightsSizePerChannel, weightsandbiases.data() + offset);
        offset += m_layerParameters.m_channels*m_weightsSizePerChannel;
        readFromBuffer(m_clContext.getCommandQueue(), m_weights[i].m_im, sizeof(cmn::GPUFLOAT)*m_layerParameters.m_channels*m_weightsSizePerChannel, weightsandbiases.data() + offset);
        offset += m_layerParameters.m_channels*m_weightsSizePerChannel;
    }
    //read biases
    readFromBuffer(m_clContext.getCommandQueue(), m_biases.m_re, m_layerParameters.m_numOfKernels*sizeof(cmn::GPUFLOAT), weightsandbiases.data() + offset);
    offset += m_layerParameters.m_numOfKernels;
    readFromBuffer(m_clContext.getCommandQueue(), m_biases.m_im, m_layerParameters.m_numOfKernels*sizeof(cmn::GPUFLOAT), weightsandbiases.data() + offset);
}

void ConvLayer::restore()
{
    std::vector<cmn::GPUFLOAT> ungroupedweights;
    m_persistance.restoreFloatBlob(getId(), ungroupedweights);
    if(ungroupedweights.empty())
    {
        BOOST_LOG_TRIVIAL(debug) << "ConvLayer, generating weights vector";
        for(unsigned int i = 0; i < m_layerParameters.m_numOfKernels; ++i)
        {
            std::vector<cmn::GPUFLOAT> real = generateRandomVector(m_layerParameters.m_channels*m_weightsSizePerChannel, m_layerParameters.m_weightsMean, m_layerParameters.m_initWeightsDeviation);
            std::vector<cmn::GPUFLOAT> imag(m_layerParameters.m_channels*m_weightsSizePerChannel, 0.0);
            if(m_layerParameters.m_weightsType == conf::ConvLayerParams::WeightsType::COMPLEX)
            {
                imag = generateRandomVector(m_layerParameters.m_channels*m_weightsSizePerChannel, m_layerParameters.m_weightsMean, m_layerParameters.m_initWeightsDeviation);
            }
            ungroupedweights.insert(ungroupedweights.end(), real.begin(), real.end());
            ungroupedweights.insert(ungroupedweights.end(), imag.begin(), imag.end());
        }
        std::vector<cmn::GPUFLOAT> biasReal(m_layerParameters.m_numOfKernels, m_layerParameters.m_initBias.real());
        std::vector<cmn::GPUFLOAT> biasIm(m_layerParameters.m_numOfKernels, m_layerParameters.m_initBias.imag());
        ungroupedweights.insert(ungroupedweights.end(), biasReal.begin(), biasReal.end());
        ungroupedweights.insert(ungroupedweights.end(), biasIm.begin(), biasIm.end());
    }
    assert((ungroupedweights.size() == 2*m_layerParameters.m_numOfKernels*(m_layerParameters.m_channels*m_weightsSizePerChannel+1)) && "ConvLayer::restore() doesn't work!");
    init(ungroupedweights);
}

void ConvLayer::setInput(gpu::BufferIO input)
{
    prepareBuffers(input);
    planForwardExecution();
}

void ConvLayer::planForwardExecution()
{
    //Bkp input
    //Calculate convolution size
    //For output kernel
    //  For each input channel calculate convolution
    //  Sum result of convolution for each channel and bias
    //  Run throgh activation function and store derivation in deltas, in bkp this will be multiplied with what comes from the right side
    //  Store output into m_io
    //End for

    planCopyBufferFwd(m_io.m_reShMem, m_layerInput.m_re, m_layerParameters.m_channels*m_inputChannelSize*sizeof(cmn::GPUFLOAT));
    planCopyBufferFwd(m_io.m_imShMem, m_layerInput.m_im, m_layerParameters.m_channels*m_inputChannelSize*sizeof(cmn::GPUFLOAT));

    //gpuPrintMatrix(*this, m_clContext.getCommandQueue(), m_layerInput.m_re, 28, 28);

    gpu::OpenCLKernelParameters kernel(m_outputWidth, m_outputWidth, 0, 0);
    for(unsigned int outKernel = 0; outKernel < m_layerParameters.m_numOfKernels; ++outKernel)
    {
        //gpuPrintMatrix(*this, m_clContext.getCommandQueue(), m_weights[outKernel].m_re, 5, 5);
        planFwd("compconv", kernel, m_layerInput.m_re, m_layerInput.m_im, m_inputChannelSize,
                                       m_weights[outKernel].m_re, m_weights[outKernel].m_im, m_layerParameters.m_kernelSize*m_layerParameters.m_kernelSize,
                                       m_io.m_reShMem, m_io.m_imShMem, m_outputChannelSize,
                                       m_layerParameters.m_inputSize, m_layerParameters.m_kernelSize,
                                       outKernel, m_layerParameters.m_channels, m_layerParameters.m_stride);

        planFwd(m_layerParameters.m_activationFunction.c_str(), kernel, m_io.m_reShMem, m_io.m_imShMem, m_outputChannelSize,
                                       m_io.m_reShMem, m_io.m_imShMem, m_outputChannelSize,
                                       m_biases.m_re, m_biases.m_im,
                                       m_deltas.m_re, m_deltas.m_im,
                                       outKernel);

    }    
}


gpu::BufferIO ConvLayer::getOutput()
{
    return m_io;
}


void ConvLayer::setBkpInput(gpu::BufferIO input)
{
    prepareBuffers(input);
    planBackwardExecution();
}

void ConvLayer::planBackwardExecution()
{
    //Calculate deltas for this layer
    calculateDeltas();

    //Calculate valid crosscorrelation of rot180(input) with sigmas from step1 to obtain the weights update
    rotateInput();

    for(unsigned int outKernel = 0; outKernel < m_layerParameters.m_numOfKernels; ++outKernel)
    {
        calculateErrors(outKernel);
        updateWeights(outKernel);
    }

}
void ConvLayer::updateWeights(const unsigned int outKernel)
{
    gpu::OpenCLKernelParameters kernel(m_layerParameters.m_kernelSize, m_layerParameters.m_kernelSize, 0, 0);
    planBck("updateWeights", kernel, m_layerInput.m_re, m_layerInput.m_im, m_inputChannelSize,  //input
                                     m_deltas.m_re, m_deltas.m_im, m_outputChannelSize,         //kernel
                                     m_weights[outKernel].m_re, m_weights[outKernel].m_im, m_weightsSizePerChannel,   //output
                                     m_biases.m_re, m_biases.m_im,
                                     m_layerParameters.m_inputSize, m_outputWidth,
                                     outKernel, m_layerParameters.m_channels, m_layerParameters.m_stride);
}

void ConvLayer::calculateErrors(const unsigned int outKernel)
{
    gpu::OpenCLKernelParameters kernel(m_layerParameters.m_inputSize, m_layerParameters.m_inputSize, 0, 0);
    //Calculate deltas to pass to layer left
    planBck("calculateErrors", kernel, m_deltas.m_re, m_deltas.m_im, m_outputChannelSize,
                                       m_weights[outKernel].m_re, m_weights[outKernel].m_im, m_weightsSizePerChannel,
                                       m_io.m_reShMem, m_io.m_imShMem,
                                       m_outputWidth, m_layerParameters.m_kernelSize,
                                       outKernel, m_layerParameters.m_channels, m_layerParameters.m_stride);
}

void ConvLayer::rotateInput()
{
    for(unsigned int inChannel = 0; inChannel < m_layerParameters.m_channels; ++inChannel)
    {
        gpu::OpenCLKernelParameters kernel1(m_layerParameters.m_inputSize/2, m_layerParameters.m_inputSize, 0, 0);
        planBck("flipdim", kernel1, m_layerInput.m_re, m_layerInput.m_im, inChannel, 1);

        gpu::OpenCLKernelParameters kernel2(m_layerParameters.m_inputSize, m_layerParameters.m_inputSize/2, 0, 0);
        planBck("flipdim", kernel2, m_layerInput.m_re, m_layerInput.m_im, inChannel, 2);
    }
}

void ConvLayer::calculateDeltas()
{
    gpu::OpenCLKernelParameters kernel(m_outputWidth, m_outputWidth, 0, 0);
    planBck("calculateLocalGradient", kernel, m_deltas.m_re, m_deltas.m_im, m_outputChannelSize,
                                              m_io.m_reShMem, m_io.m_imShMem, m_outputChannelSize,
                                              m_layerParameters.m_numOfKernels);
}

gpu::BufferIO ConvLayer::getBkpOutput()
{
    return m_io;
}

void ConvLayer::init(std::vector<cmn::GPUFLOAT>& rawdata)
{
    //init weights
    for(unsigned int i = 0; i < m_layerParameters.m_numOfKernels; ++i)
    {
        BOOST_LOG_TRIVIAL(debug) << "ConvLayer for kernel " << i << " initializing weights for im and re";
        m_weights.emplace_back(cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , m_layerParameters.m_channels*m_weightsSizePerChannel*sizeof(cmn::GPUFLOAT), (void*)(rawdata.data() + 2*i*m_layerParameters.m_channels*m_weightsSizePerChannel)),
                               cl::Buffer(m_clContext.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , m_layerParameters.m_channels*m_weightsSizePerChannel*sizeof(cmn::GPUFLOAT), (void*)(rawdata.data() + (2*i+1)*m_layerParameters.m_channels*m_weightsSizePerChannel)),
                               m_layerParameters.m_channels*m_weightsSizePerChannel);
    }

    //init biases
    cmn::GPUFLOAT* offset = rawdata.data() + 2*m_layerParameters.m_numOfKernels*m_layerParameters.m_channels*m_weightsSizePerChannel;
    writeToBuffer(m_clContext.getCommandQueue(), m_biases.m_re, m_layerParameters.m_numOfKernels*sizeof(cmn::GPUFLOAT), offset);
    writeToBuffer(m_clContext.getCommandQueue(), m_biases.m_im, m_layerParameters.m_numOfKernels*sizeof(cmn::GPUFLOAT), offset+m_layerParameters.m_numOfKernels);
}

void ConvLayer::prepareBuffers(gpu::BufferIO input)
{
    m_io = input;
}
