#include <SpectralPoolingLayer.h>
#include <OpenCLContext.h>
#include <InputLayer.h>
#include <boost/log/trivial.hpp>
#include "Utils.h"

using namespace neneta;
using namespace neneta::net;

SpectralPoolingLayer::SpectralPoolingLayer(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext)
    : gpu::OpenCLExecutionPlan(layerId, confReader, program)
    , m_layerParameters(conf::PoolingLayerConfiguration(getId(), confReader).getParamSet(getId()))
    , m_oclKernelParameters(m_layerParameters.m_outputSize, m_layerParameters.m_outputSize, getKernelConfiguration().getLWS(1), getKernelConfiguration().getLWS(2))
    , m_clContext(oclContext)
    , m_clProgram(program)
    , m_tempBuffers()
    , m_reTmp(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT))
    , m_imTmp(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_outputSize*m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT))
{
}

SpectralPoolingLayer::~SpectralPoolingLayer()
{
}

void SpectralPoolingLayer::getRe(std::vector<std::vector<cmn::GPUFLOAT>>& re)
{
    cmn::GPUFLOAT* data = new cmn::GPUFLOAT[m_layerParameters.m_outputSize*m_layerParameters.m_outputSize];
 //   memset(data, 0, m_layerParameters.m_outputSize*m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT));
    for(unsigned int channel = 0; channel < m_layerParameters.m_channels; ++channel)
    {
        re.emplace_back();
        readFromBuffer(m_clContext.getCommandQueue(), m_io.m_reChannels[channel], m_layerParameters.m_outputSize*m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT), data);
        re.back().assign(data, data + m_layerParameters.m_outputSize*m_layerParameters.m_outputSize);
    }
    delete [] data;
}

void SpectralPoolingLayer::getIm(std::vector<std::vector<cmn::GPUFLOAT>>& im)
{
    cmn::GPUFLOAT* data = new cmn::GPUFLOAT[m_layerParameters.m_outputSize*m_layerParameters.m_outputSize];
  //  memset(data, 0, m_layerParameters.m_outputSize*m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT));
    for(unsigned int channel = 0; channel < m_layerParameters.m_channels; ++channel)
    {
        im.emplace_back();
        readFromBuffer(m_clContext.getCommandQueue(), m_io.m_imChannels[channel], m_layerParameters.m_outputSize*m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT), data);
        im.back().assign(data, data + m_layerParameters.m_outputSize*m_layerParameters.m_outputSize);
    }
    delete [] data;
}

void SpectralPoolingLayer::setInput(gpu::BufferIO input)
{
    try
    {
        prepareBuffers(input);

        unsigned int newSpectralSizeBytes = m_layerParameters.m_outputSize*m_layerParameters.m_outputSize*sizeof(cmn::GPUFLOAT);

        for(unsigned int ch = 0; ch < m_layerParameters.m_channels; ++ch)
        {
            cl_buffer_region bg{ch*newSpectralSizeBytes, newSpectralSizeBytes};
            m_io.m_reChannels.emplace_back(m_io.m_reShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));
            m_io.m_imChannels.emplace_back(m_io.m_imShMem.createSubBuffer(CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &bg));

            planFwd("spectral_pooling", m_oclKernelParameters, m_reTmp, m_imTmp, m_tempBuffers.m_reChannels[ch], m_tempBuffers.m_imChannels[ch],
                                     m_layerParameters.m_outputSize, m_layerParameters.m_inputSize);

            planCopyBufferFwd(m_reTmp, m_io.m_reChannels.back(),
                       sizeof(cmn::GPUFLOAT)*m_layerParameters.m_outputSize*m_layerParameters.m_outputSize);

            planCopyBufferFwd(m_imTmp, m_io.m_imChannels.back(),
                       sizeof(cmn::GPUFLOAT)*m_layerParameters.m_outputSize*m_layerParameters.m_outputSize);

        }
    }
    catch(const cl::Error& err)
    {
        BOOST_LOG_TRIVIAL(debug) << "Exception in SpectralPoolingLayer err " << err.what() << " id = " << err.err();
        throw;
    }
}

gpu::BufferIO SpectralPoolingLayer::getOutput()
{
    return m_io;
}

void SpectralPoolingLayer::setBkpInput(gpu::BufferIO input)
{
    std::string str;
    str += "SpectralPoolingLayer id=" + getId() + std::string(" setBkpInput");
    gpu::OpenCLKernelParameters kparams(256,256);
    writeToBuffer(m_clContext.getCommandQueue(), m_reTmp, str.size(), const_cast<char*>(str.data()));
    planBck("println", kparams, m_reTmp, (cl_int)str.size());
}

gpu::BufferIO SpectralPoolingLayer::getBkpOutput()
{
    return m_io;
}

void SpectralPoolingLayer::prepareBuffers(gpu::BufferIO input)
{
    m_io = input;
    m_tempBuffers.clear();
    m_tempBuffers.m_reChannels.swap(m_io.m_reChannels);
    m_tempBuffers.m_imChannels.swap(m_io.m_imChannels);
}
