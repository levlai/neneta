#pragma once

#include <Configuration.h>
#include <LayerConfiguration.h>
#include <string>
#include <Types.h>

namespace neneta
{
namespace conf
{

struct FFTConvLayerParams : public LayerParams
{
    FFTConvLayerParams(unsigned int channels, unsigned int kernels, unsigned int kernelsize, unsigned int inputSize, cmn::GPUFLOAT weightsdev, cmn::GPUFLOAT bias, const std::string& id)
        : LayerParams(ELayerType::SPECTRAL_CONV)
        , m_channels(channels)
        , m_numOfKernels(kernels)
        , m_kernelSize(kernelsize)
        , m_inputSize(inputSize)
        , m_initWeightsDeviation(weightsdev)
        , m_initBias(bias)
        , m_id(id)
    {}
    unsigned int m_channels;
    unsigned int m_numOfKernels;
    unsigned int m_kernelSize;
    unsigned int m_inputSize;
    cmn::GPUFLOAT        m_initWeightsDeviation;
    cmn::GPUFLOAT        m_initBias;
    std::string  m_id;
};


class ConfigurationReader;

class FFTConvLayerConfiguration : public Configuration<FFTConvLayerParams>
{
public:
    FFTConvLayerConfiguration(const std::string& layerId, const ConfigurationReader& confReader);
};

}
}
