#include <InverseFourierTransform.h>
#include <OpenCLContext.h>
#include <boost/log/trivial.hpp>

using namespace neneta;
using namespace neneta::net;

InverseFourierTransform::InverseFourierTransform(const std::string& layerId, const conf::ConfigurationReader& confReader, const gpu::OpenCLProgram& program, const gpu::OpenCLContext& oclContext)
    : gpu::OpenCLExecutionPlan(layerId, confReader, program)
    , m_layerParameters(conf::FourierConfiguration(layerId, confReader).getParamSet(layerId))
    , m_oclKernelParameters(m_layerParameters.m_size, m_layerParameters.m_size, getKernelConfiguration().getLWS(1), getKernelConfiguration().getLWS(2))
    , m_clContext(oclContext)
    , m_io()
    , m_reTmp(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_size*m_layerParameters.m_size*sizeof(cmn::GPUFLOAT))
    , m_imTmp(m_clContext.getContext(), CL_MEM_READ_WRITE, m_layerParameters.m_size*m_layerParameters.m_size*sizeof(cmn::GPUFLOAT))
{
}

InverseFourierTransform::~InverseFourierTransform()
{
}

void InverseFourierTransform::init()
{
    for(unsigned int channel = 0; channel < m_layerParameters.m_channels; ++channel)
    {
        // plan("fftShift", {m_layerParameters.m_size, m_layerParameters.m_size, m_layerParameters.m_size/2, m_layerParameters.m_size/2}, *reKernelIt, *imKernelIt, (cl_int)m_layerParameters.m_size);
        planFwd("rowIndexReverse", m_oclKernelParameters, m_io.m_reChannels[channel], m_io.m_imChannels[channel], (cl_int)m_layerParameters.m_size, (cl_int)std::log2(m_layerParameters.m_size));
        planFwd("columnIndexReverse", m_oclKernelParameters, m_io.m_reChannels[channel], m_io.m_imChannels[channel], (cl_int)m_layerParameters.m_size, (cl_int)std::log2(m_layerParameters.m_size));
        for(int stage = 1; stage <= std::log2(m_layerParameters.m_size); ++stage)
        {
           planFwd("dit_2x2radix_2dfft_1st", m_oclKernelParameters, m_io.m_reChannels[channel], m_io.m_imChannels[channel], m_reTmp, m_imTmp, (cl_int)m_layerParameters.m_size, (cl_int)stage, (cl_int)1);
           planFwd("dit_2x2radix_2dfft_2nd", m_oclKernelParameters, m_io.m_reChannels[channel], m_io.m_imChannels[channel], m_reTmp, m_imTmp, (cl_int)m_layerParameters.m_size, (cl_int)stage);
        }
        planFwd("fftScale", m_oclKernelParameters, m_io.m_reChannels[channel], m_io.m_imChannels[channel], (cl_int)m_layerParameters.m_size);
    }
}

void InverseFourierTransform::getRe(std::vector<std::vector<cmn::GPUFLOAT>>& re)
{
    cmn::GPUFLOAT* data = new cmn::GPUFLOAT[m_layerParameters.m_size*m_layerParameters.m_size];
    for(unsigned int channel = 0; channel < m_layerParameters.m_channels; ++channel)
    {
        re.emplace_back();
        readFromBuffer(m_clContext.getCommandQueue(), m_io.m_reChannels[channel], m_layerParameters.m_size*m_layerParameters.m_size*sizeof(cmn::GPUFLOAT), data);
        re.back().assign(data, data + m_layerParameters.m_size*m_layerParameters.m_size);
    }
    delete [] data;
}

void InverseFourierTransform::getIm(std::vector<std::vector<cmn::GPUFLOAT>>& im)
{
    cmn::GPUFLOAT* data = new cmn::GPUFLOAT[m_layerParameters.m_size*m_layerParameters.m_size];
    for(unsigned int channel = 0; channel < m_layerParameters.m_channels; ++channel)
    {
        im.emplace_back();
        readFromBuffer(m_clContext.getCommandQueue(), m_io.m_imChannels[channel], m_layerParameters.m_size*m_layerParameters.m_size*sizeof(cmn::GPUFLOAT), data);
        im.back().assign(data, data + m_layerParameters.m_size*m_layerParameters.m_size);
    }
    delete [] data;
}


void neneta::net::InverseFourierTransform::setInput(gpu::BufferIO input)
{
    m_io = input;
    init();
}

gpu::BufferIO neneta::net::InverseFourierTransform::getOutput()
{
    return m_io;
}

void InverseFourierTransform::setBkpInput(gpu::BufferIO input)
{

}

gpu::BufferIO InverseFourierTransform::getBkpOutput()
{
    return m_io;
}
