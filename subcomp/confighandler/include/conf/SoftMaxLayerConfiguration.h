#pragma once

#include <Configuration.h>
#include <LayerConfiguration.h>
#include <string>
#include <Types.h>

namespace neneta
{
namespace conf
{

struct SoftMaxParams : public LayerParams
{
    SoftMaxParams(unsigned int channels,
                  unsigned int inputDim,
                  unsigned int inputSize,
                  unsigned int outputSize,
                  const std::string& actFunc,
                  cmn::GPUFLOAT weightsdev,
                  cmn::GPUFLOAT weightsmean,
                  cmn::GPUFLOAT bias,
                  const std::string& id)
        : LayerParams(ELayerType::SOFT_MAX)
        , m_channels(channels)
        , m_inputDim(inputDim)
        , m_inputSize(inputSize)
        , m_outputSize(outputSize)
        , m_activationFunction(actFunc)
        , m_initWeightsDeviation(weightsdev)
        , m_weightsMean(weightsmean)
        , m_initBias(bias)
        , m_id(id)
    {}
    unsigned int m_channels; 
    unsigned int m_inputDim;
    unsigned int m_inputSize;
    unsigned int m_outputSize;
    std::string  m_activationFunction;
    cmn::GPUFLOAT m_initWeightsDeviation;
    cmn::GPUFLOAT m_weightsMean;
    cmn::GPUFLOAT m_initBias;
    std::string  m_id;
};


class ConfigurationReader;

class SoftMaxLayerConfiguration : public Configuration<SoftMaxParams>
{
public:
    SoftMaxLayerConfiguration(const std::string& id, const ConfigurationReader& confReader);

};

}
}
