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

struct FCParams : public LayerParams
{
    enum WeightsType
    {
        REAL,
        COMPLEX
    };

    FCParams(unsigned int channels,
             unsigned int inputDim,
             unsigned int inputSize,
             unsigned int outputSize,
             cmn::GPUFLOAT weightsdev,
             cmn::GPUFLOAT weightsmean,
             WeightsType wt,
             cmn::GPUFLOAT biasRe,
             cmn::GPUFLOAT biasIm,
             const std::string& actFunc,
             const std::string& id)
        : LayerParams(ELayerType::FULLY_CONN)
        , m_channels(channels)
        , m_inputDim(inputDim)
        , m_inputSize(inputSize)
        , m_outputSize(outputSize)
        , m_initWeightsDeviation(weightsdev)
        , m_weightsMean(weightsmean)
        , m_weightsType(wt)
        , m_initBias(biasRe, biasIm)
        , m_activationFunction(actFunc)
        , m_id(id)
    {}
    unsigned int m_channels;
    unsigned int m_inputDim;
    unsigned int m_inputSize;
    unsigned int m_outputSize;
    cmn::GPUFLOAT        m_initWeightsDeviation;
    cmn::GPUFLOAT m_weightsMean;
    WeightsType  m_weightsType;
    std::complex<cmn::GPUFLOAT>  m_initBias;
    std::string  m_activationFunction;
    std::string  m_id;
};


class ConfigurationReader;

class FCLayerConfiguration : public Configuration<FCParams>
{
public:
    FCLayerConfiguration(const std::string& layerId, const ConfigurationReader& confReader);

};

}
}
