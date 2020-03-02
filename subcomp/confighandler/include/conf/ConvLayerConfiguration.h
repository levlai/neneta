#pragma once

#include <Configuration.h>
#include <LayerConfiguration.h>
#include <string>
#include <complex>
#include <Types.h>

namespace neneta
{
namespace conf
{

struct ConvLayerParams : public LayerParams
{
    enum WeightsType
    {
        REAL,
        COMPLEX
    };

    ConvLayerParams(unsigned int channels,
                    unsigned int kernels,
                    unsigned int kernelsize,
                    unsigned int stride,
                    unsigned int inputDim,
                    unsigned int inputSize,
                    cmn::GPUFLOAT weightsdev,
                    cmn::GPUFLOAT weightsmean,
                    WeightsType wt,
                    cmn::GPUFLOAT biasre,
                    cmn::GPUFLOAT biasim,
                    const std::string& actFunc,
                    const std::string& id)
        : LayerParams(ELayerType::SPECTRAL_CONV)
        , m_channels(channels)
        , m_numOfKernels(kernels)
        , m_kernelSize(kernelsize)
        , m_stride(stride)
        , m_inputDim(inputDim)
        , m_inputSize(inputSize)
        , m_initWeightsDeviation(weightsdev)
        , m_weightsMean(weightsmean)
        , m_weightsType(wt)
        , m_initBias(biasre, biasim)
        , m_activationFunction(actFunc)
        , m_id(id)
    {}
    unsigned int m_channels;
    unsigned int m_numOfKernels;
    unsigned int m_kernelSize;
    unsigned int m_stride;
    unsigned int m_inputDim;
    unsigned int m_inputSize;
    cmn::GPUFLOAT        m_initWeightsDeviation;
    cmn::GPUFLOAT m_weightsMean;
    WeightsType  m_weightsType;
    std::complex<cmn::GPUFLOAT>  m_initBias;
    std::string  m_activationFunction;
    std::string  m_id;
};


class ConfigurationReader;

class ConvLayerConfiguration : public Configuration<ConvLayerParams>
{
public:
    ConvLayerConfiguration(const std::string& layerId, const ConfigurationReader& confReader);
};

}
}
