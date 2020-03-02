#include <FFTConvLayer.h>
#include <OpenCLContext.h>
#include <Image.h>
#include <InputLayer.h>
#include "Utils.h"

using namespace neneta;
using namespace neneta::net;

FFTConvLayer::FFTConvLayer(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext)
    : gpu::OpenCLExecutionPlan(layerId, confReader, program)
    , m_persistance(confReader)
    , m_layerParameters(conf::FFTConvLayerConfiguration(getId(), confReader).getParamSet(getId()))
    , m_oclKernelParameters(m_layerParameters.m_inputSize, m_layerParameters.m_inputSize, getKernelConfiguration().getLWS(1), getKernelConfiguration().getLWS(2))
    , m_clContext(oclContext)
    , m_clProgram(program)
    , m_tempBuffers()
    , m_reTmp(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_inputSize*m_layerParameters.m_inputSize*sizeof(cmn::GPUFLOAT))
    , m_imTmp(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_inputSize*m_layerParameters.m_inputSize*sizeof(cmn::GPUFLOAT))
{
    restore();
    init(confReader);
}

FFTConvLayer::~FFTConvLayer()
{
}

void FFTConvLayer::store()
{
    //convert fftKernels to clKernels and store to db
}

void FFTConvLayer::restore()
{
    //read from db and load in m_kernels
    //one kernel in the layer should have input_channels*kernelSize^2*sizeof(cmn::GPUFLOAT) bytes
    m_persistance.restoreFloatBlob(getId(), m_kernels);
    if(m_kernels.empty())
    {
        BOOST_LOG_TRIVIAL(debug) << "FFTConvLayer, generating layer parameters.";
        m_kernels = generateRandomVector(m_layerParameters.m_channels*m_layerParameters.m_kernelSize*m_layerParameters.m_kernelSize*m_layerParameters.m_numOfKernels, 0.0, m_layerParameters.m_initWeightsDeviation);
    /*   m_kernels = {-1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1};*/
       // m_kernels = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0}; // do nothing
    }
}

// m_numOfKernels is here new number of channels
//Kernels in m_fftKernels vec<list> are stored like:
// vec[ch1] = list(kn11,..,kn1M)
// ..
// vec[chN] = list(knN1,..,knNM)
//
//m_kernels is a cmn::GPUFLOAT vector of kernels where kernels are stored like:
// [kn11,..,kn1M,kn21,..,kn2M,..,knN1,..knNM] - channel ordered
void FFTConvLayer::setInput(gpu::BufferIO input)
{
    prepareBuffers(input);

    unsigned int spectralSizeBytes = m_layerParameters.m_inputSize*m_layerParameters.m_inputSize*sizeof(cmn::GPUFLOAT);

    for(unsigned int newChannel = 0; newChannel < m_layerParameters.m_numOfKernels; ++newChannel)
    {

        cl_buffer_region bg{newChannel*spectralSizeBytes, spectralSizeBytes};
        m_io.m_reChannels.emplace_back(m_io.m_reShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));
        m_io.m_imChannels.emplace_back(m_io.m_imShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));

        //create temp buffer for re and im
        //initialize it to zero
        //change hadarmard to be +=
        //copy the temp buffer after to new m_io.m_reChannels
        for(unsigned int inChannel = 0; inChannel < m_layerParameters.m_channels; ++inChannel)
        {
            gpu::BufferIO fftKernelBuffers = m_fftKernels[inChannel][newChannel].getOutput();

            planFwd("hadamard_product", m_oclKernelParameters, m_reTmp, m_imTmp,
                                     m_tempBuffers.m_reChannels[inChannel], m_tempBuffers.m_imChannels[inChannel],
                                     fftKernelBuffers.m_reChannels.front(), fftKernelBuffers.m_imChannels.front(),
                                     m_layerParameters.m_inputSize, (cl_int)inChannel);

        }
        //plan("add bias")...

        planCopyBufferFwd(m_reTmp, m_io.m_reChannels.back(),
                   sizeof(cmn::GPUFLOAT)*m_layerParameters.m_inputSize*m_layerParameters.m_inputSize);
        planCopyBufferFwd(m_imTmp, m_io.m_imChannels.back(),
                   sizeof(cmn::GPUFLOAT)*m_layerParameters.m_inputSize*m_layerParameters.m_inputSize);
    }
}

gpu::BufferIO FFTConvLayer::getOutput()
{
    return m_io;
}


void FFTConvLayer::setBkpInput(gpu::BufferIO input)
{
    std::string str;
    str += "FFTConvLayer id=" + getId() + std::string(" setBkpInput");
    gpu::OpenCLKernelParameters kparams(256,256);
    writeToBuffer(m_clContext.getCommandQueue(), m_reTmp, str.size(), const_cast<char*>(str.data()));
    planBck("println", kparams, m_reTmp, (cl_int)str.size());
}

gpu::BufferIO FFTConvLayer::getBkpOutput()
{
    return m_io;
}

void FFTConvLayer::init(const conf::ConfigurationReader& confReader)
{
    conf::FourierParams fftParams("", m_layerParameters.m_inputSize, 1, "FFTConvLayerFourier");
    conf::InputLayerParams inputParams(m_layerParameters.m_inputSize*m_layerParameters.m_inputSize*sizeof(cmn::GPUFLOAT),
                                       m_layerParameters.m_inputSize*m_layerParameters.m_inputSize*sizeof(cmn::GPUFLOAT),
                                       2, 256, 1, 3, "FFTConvLayerInput", conf::LabelCoding::REAL);

    const auto kernelSize = m_layerParameters.m_kernelSize;
    const auto kernelOffset = kernelSize*kernelSize;

    BOOST_LOG_TRIVIAL(debug) << "FFTConvLayer fft of weights";
    for(unsigned int inChannel = 0; inChannel < m_layerParameters.m_channels; ++inChannel)
    {
        BOOST_LOG_TRIVIAL(debug) << "For input channel " << inChannel << "...";
        m_fftKernels.emplace_back();
        for(unsigned int newChannel = 0; newChannel < m_layerParameters.m_numOfKernels; ++newChannel)
        {
            BOOST_LOG_TRIVIAL(debug) << "adding new channel " << newChannel << " of size " << kernelSize;
            gpu::IOpenCLInputExecutionPlan::ImageType kernel(m_kernels.data() + (inChannel*m_layerParameters.m_numOfKernels+newChannel)*kernelOffset, kernelSize);
            kernel.zeroPad(m_layerParameters.m_inputSize - kernelSize);
            net::InputLayer input(inputParams, confReader, m_clProgram, m_clContext);
            m_fftKernels[inChannel].emplace_back(fftParams, confReader, m_clProgram, m_clContext);

            input.setInput(kernel);
            input >> m_fftKernels[inChannel][newChannel];

            m_fftKernels[inChannel][newChannel].runFwdPropagation(m_clContext);
            m_fftKernels[inChannel][newChannel].wait();
        }
    }
}

#include <limits.h>
void FFTConvLayer::prepareBuffers(gpu::BufferIO input)
{
    m_io = input;
    m_tempBuffers.clear();
    m_tempBuffers.m_reChannels.swap(m_io.m_reChannels);
    m_tempBuffers.m_imChannels.swap(m_io.m_imChannels);
}
